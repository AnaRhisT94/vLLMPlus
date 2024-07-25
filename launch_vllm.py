#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from queue import Queue
import json
from typing import List, Dict, Optional
import time
import asyncio
import argparse
from types import SimpleNamespace
import subprocess
import threading
from dataclasses import dataclass
#from vllm import ServerArgs, LLMServer, SamplingParams
from vllm import SamplingParams, EngineArgs, LLMEngine
from vllm.utils import FlexibleArgumentParser
os.environ['HF_HOME'] = '/home/ilan/.cache/huggingface/hub/'
# assert 'CUDA_VISIBLE_DEVICES' in os.environ, "Set CUDA_VISIBLE_DEVICES, else this will take memory on each (and load model to 0)"
import torch
from typing import Annotated
from fastapi import FastAPI, Form

app = FastAPI()

from pydantic import BaseModel
class Item(BaseModel):
    inputs: List[str] = ['Hello world']
    parameters: Dict[str, int] = {'reponse_len': 10,
                                  }

@dataclass
class GenerationInputs:
    req_id: int
    prompt: str
    sampling_config: dict


@dataclass
class GenerationOutput:
    req_id: int
    generated_text: str
    num_output_tokens: int
    error: str


class ModelThread:
    """
    A class representing a thread for running a VLLM model.

    Args:
        vllm_args (list): The arguments for initializing the VLLM model.
        model_ready_event (threading.Event): An event to signal when the model is ready.
        progress_call (function): A function to call for reporting progress.
        loop (asyncio.AbstractEventLoop): The event loop to run the progress call in.

    Attributes:
        vllm_args (list): The arguments for initializing the VLLM model.
        model_ready_event (threading.Event): An event to signal when the model is ready.
        thread (threading.Thread): The thread object for running the model.
        input_queue (Queue): A queue for storing input data.
        output_queue (Queue): A queue for storing output data.
        progress_call (function): A function to call for reporting progress.
        loop (asyncio.AbstractEventLoop): The event loop to run the progress call in.
    """

    def __init__(self, vllm_args, model_ready_event, progress_call, loop):
        self.vllm_args = vllm_args
        self.model_ready_event = model_ready_event
        self.thread = None
        self.input_queue = Queue()
        self.output_queue = Queue()

        self.progress_call = progress_call
        self.loop = loop

    def start_thread(self):
        """
        Starts the thread for running the VLLM model.
        """
        self.thread = threading.Thread(target=self._thread, daemon=True)
        self.thread.start()

    def _thread(self):
        """
        The main thread function for running the VLLM model.
        The step method works as follows:

        1.) It schedules the sequences to be executed in the next iteration and the token blocks to be swapped in/out/copy.
            Sequences may be preempted or reordered based on the scheduling policy.

        2.) It calls the distributed executor to execute the model.

        3.) It processes the model output, which includes decoding the relevant outputs, updating the scheduled sequence groups with model outputs
            based on its sampling parameters, and freeing the finished sequence groups.

        4.) Finally, it creates and returns the newly generated results.

        The step method also handles logging of stats and tracing, and it stops the execution loop in parallel workers if there are no more unfinished requests.
        """
        server = self.init_model(self.vllm_args)
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model_ready_event.set()

        while True:
            time.sleep(0.01)

            gen_inputs = []
            while not self.input_queue.empty():
                gen_input = self.input_queue.get_nowait()
                reponse_len = gen_input.sampling_config['response_len']
                prompt = {'prompt': gen_input.prompt}
                sampling_params = SamplingParams(
                    n=1,
                    # yeah, typo
                    max_tokens=max(
                        reponse_len, 1),
                    ignore_eos=True,
                )
                req_id = gen_input.req_id

                server.add_request(
                    str(req_id),
                    prompt,
                    sampling_params,
                )

            vllm_outputs = server.step()

            needs_call_progress = False
            for vllm_output in vllm_outputs:
                if not vllm_output.finished:
                    continue

                needs_call_progress = True
                assert len(vllm_output.outputs) == 1
                req_id = int(vllm_output.request_id)
                generated_text = vllm_output.outputs[0].text
                num_output_tokens = len(vllm_output.outputs[0].token_ids)

                gen_output = GenerationOutput(
                    req_id=req_id,
                    generated_text=generated_text,
                    num_output_tokens=num_output_tokens,
                    error=None,
                )
                self.output_queue.put_nowait(gen_output)

            if needs_call_progress:
                asyncio.run_coroutine_threadsafe(self.progress_call(), loop)

    @staticmethod
    def init_model(vllm_args):
        """
        Initializes the VLLM model.

        Args:
            vllm_args (list): The arguments for initializing the VLLM model.

        Returns:
            LLMServer: The initialized VLLM server.
        """
        print('Init model')
        #server_args = ServerArgs.from_cli_args(vllm_args)
       # server = LLMServer.from_server_args(server_args)
        engine_args = EngineArgs.from_cli_args(vllm_args)
        server = LLMEngine.from_engine_args(engine_args)
        # server = LLM(model='facebook/opt-125m',trust_remote_code=True, dtype=torch.bfloat16,
        #              gpu_memory_utilization=0.4)
        print('Model ready')
        return server


class FastAPIServer:
    """
    A class representing a FastAPI server for generating text using a VLLM model.

    Attributes:
        model_ready_event (asyncio.Event): An event indicating whether the model is ready.
        requests (dict): A dictionary to store the requests made to the server.
        generations (dict): A dictionary to store the generated text and related information.
        request_queue (list): A list to store the request IDs in the order they are received.
        _next_req_id (int): An integer representing the next request ID.
        loop (asyncio.AbstractEventLoop): The event loop for handling asynchronous tasks.
        model_thread (ModelThread): An instance of the ModelThread class for running the VLLM model.

    Methods:
        next_req_id: Get the next request ID.
        progress_async: Asynchronously update the progress of the server.
        progress: Update the progress of the server.
        is_ready: Check if the model is ready.
        add_request: Add a new request to the server.
        get_generation: Get the generated text for a specific request.
        generate: Generate text based on the given request dictionary.

    """

    def __init__(self, loop, vllm_args):
        self.model_ready_event = asyncio.Event()

        self.requests = {}
        self.generations = {}
        self.request_queue = []
        self._next_req_id = 0

        self.loop = loop

        self.model_thread = ModelThread(
            vllm_args, self.model_ready_event, self.progress_async, self.loop)
        self.model_thread.start_thread()

    @property
    def next_req_id(self):
        rval = self._next_req_id
        self._next_req_id += 1
        return rval

    async def progress_async(self):
        return self.progress()

    def progress(self):
        '''
        The progress method is responsible for sending requests from the request queue to the model and receiving the generated outputs from the model.
        It first iterates over each request in the request queue, retrieves the corresponding prompt and configuration, packages these into a GenerationInputs object,
        and puts this object into the model's input queue. It then empties the request queue and checks the model's output queue for any generated outputs,
        appending any found outputs to a list. 
        Finally, it updates the generations dictionary with the generated outputs and sets the corresponding ready_event for each output.
        '''
        sent_to_model = 0
        recv_from_model = 0

        for req_id in self.request_queue:
            prompt, sampling_config = self.requests[req_id]
            gen_inputs = GenerationInputs(
                req_id,
                prompt,
                sampling_config,
            )
            self.model_thread.input_queue.put_nowait(gen_inputs)
            sent_to_model += 1
        self.request_queue = []

        found_outputs = []
        while not self.model_thread.output_queue.empty():
            gen_output = self.model_thread.output_queue.get_nowait()
            found_outputs.append(gen_output)
            recv_from_model += 1

        for output in found_outputs:
            req_id = output.req_id
            ready_event, _, _, _ = self.generations[req_id]
            self.generations[req_id] = (
                ready_event, output.generated_text, output.num_output_tokens, output.error)
            ready_event.set()

        print(f'progress {sent_to_model=} {recv_from_model=}')

    async def is_ready(self):
        return self.model_ready_event.is_set()

    def add_request(self, prompt, sampling_config):
        '''
        The add_request method adds a new request to the request queue and the requests dictionary.
        It also creates a new asyncio.Event instance, which is used to signal when the generation for this request is complete,
        and adds a new entry to the generations dictionary.
        '''
        req_id = self.next_req_id
        self.requests[req_id] = (prompt, sampling_config)
        self.request_queue.append(req_id)

        ready_event = asyncio.Event()
        self.generations[req_id] = (ready_event, None, None, None)
        return req_id

    async def get_generation(self, req_id):
        '''
        The get_generation method is an asynchronous method that waits for the ready_event of a specific request to be set,
        indicating that the generation is complete. It then retrieves the generated text, the number of output tokens,
        and any error that occurred during generation.
        It also cleans up by deleting the entries for this request from the generations and requests dictionaries.
        '''
        ready_event, _, _, _ = self.generations[req_id]
        await ready_event.wait()
        _, generation, num_output_tokens, error = self.generations[req_id]

        del self.generations[req_id]
        del self.requests[req_id]
        return generation, num_output_tokens, error

    async def generate(self, item: Item):
        '''
        The generate method is another asynchronous method that takes a dictionary containing the inputs and parameters for a generation request.
        It adds the request, triggers the progress method to process requests, and waits for the generation to complete.
        It then checks that the number of output tokens matches the expected response length and returns a dictionary containing the generated text,
        the number of output tokens, and any error that occurred.
        '''
        prompt = item['inputs']
        sampling_config = item['parameters']
        print(f"Prompt: {prompt}")
        print(f"sampling_config: {sampling_config}")
        
        req_id = self.add_request(prompt, sampling_config)
        self.progress()
        generation, num_output_tokens, error = await self.get_generation(req_id)

        expected_resp_len = sampling_config['response_len']
        # print(f'generate check_len: {num_output_tokens=} {expected_resp_len=}')
        assert max(expected_resp_len, 1) == max(num_output_tokens,
                                                1), f"{expected_resp_len=} {num_output_tokens=}"

        return {
            'generated_text': generation,
            'num_output_tokens_cf': num_output_tokens,
            'error': error,
        }

        
@app.post("/generate")
async def generate_stream(dict: Dict):
    return await server.generate(dict)


@app.get("/is_ready")
async def is_ready(request: Request):
    return await server.is_ready()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser_fastapi = argparse.ArgumentParser()
    # parser_fastapi.add_argument('--port', type=int, default=8081)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.4)
    parser.add_argument('--enforce-eager', action="store_true")
    engine_parser = FlexibleArgumentParser(parser)
    EngineArgs.add_cli_args(engine_parser)
    engine_args = engine_parser.parse_args()
    args = parser.parse_args()
    # args_fastapi = parser_fastapi.parse_args()
    
    vllm_args = EngineArgs.from_cli_args(engine_args)
    # vllm_args = None
    loop = asyncio.new_event_loop()
    server = FastAPIServer(loop, vllm_args)

    from uvicorn import Config, Server
    config = Config(app=app, loop=loop, host='localhost',
                    port=8081, log_level="info")
    uvicorn_server = Server(config)

    loop.run_until_complete(uvicorn_server.serve())
