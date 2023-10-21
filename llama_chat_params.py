# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from llama import Llama, Dialog
import http.server
import socketserver
import json
import argparse

# --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 6

temperature: float = 0.6
top_p: float = 0.9
max_gen_len: Optional[int] = None


class MyHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
        
    def _send_response(self, status_code, response_content):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_content).encode())

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        try:
            json_data = json.loads(post_data)
            request = json_data["request"]
            response = get_response(request)
            response = {"response": response}
            self._send_response(200, response)
        except ValueError as e:
            error_response = {"error": "Invalid JSON data"}
            self._send_response(400, error_response)


class LlamaGenerator:

    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
        self.generator = None
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    def generate(self):
        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )


def get_response(request="hello"):
    dialogs: List[Dialog] = [
        [{"role": "user", "content": request}],
    ]

    print("> user: " + request)

    results = llama_generator.generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for result in results:
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
    print("\n==================================\n")

    if len(results) > 0:
        return results[0]['generation']['content']

    return "I do not understand the request!!!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start llama")
    parser.add_argument("--ckpt_dir", type=str,
                        help=" The directory containing checkpoint files for the pretrained model.")
    parser.add_argument("--tokenizer_path", type=str,
                        help="The path to the tokenizer model used for text encoding/decoding.")
    parser.add_argument("--max_seq_len", type=int,
                        help="The maximum sequence length for input prompts. Defaults to 512.")
    parser.add_argument("--max_batch_size", type=int,
                        help="The maximum batch size for generating sequences. Defaults to 8.")

    args = parser.parse_args()
    llama_generator = LlamaGenerator(args.ckpt_dir, args.tokenizer_path, args.max_seq_len, args.max_batch_size)
    llama_generator.generate()

    # Set the server address and port
    server_address = ('', 8000)

    # Create and start the server
    httpd = socketserver.TCPServer(server_address, MyHandler)
    print("Server is running on port 8000...")
    httpd.serve_forever()
