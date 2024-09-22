
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGeneratorAsync, ExLlamaV2DynamicJobAsync
import asyncio

async def main():
    model_dir = "/mnt/str/models/llama3-8b-exl2/4.0bpw"
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, lazy = True)
    model.load_autosplit(cache, progress = True)

    print("Loading tokenizer...")
    tokenizer = ExLlamaV2Tokenizer(config)

    # Initialize the async generator with all default parameters

    generator = ExLlamaV2DynamicGeneratorAsync(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )

    # Run some async job tasks

    prompts = [
        "Once upon a time, there was",
        "asyncio in Python is a great feature because",
        "asyncio in Python is a pain to work with because",
    ]

    async def run_job(prompt: str, marker: str):

        # Create an iterable, asynchronous job. The job will be transparently batched together with other concurrent
        # jobs on the started on the same generator.
        job = ExLlamaV2DynamicJobAsync(
            generator,
            input_ids = tokenizer.encode(prompt, add_bos = False),
            max_new_tokens = 200
        )

        full_completion = prompt

        # Iterate through the job. Each returned result is a dictionary containing an update on the status of the
        # job and/or part of the completion (see ExLlamaV2DynamicGenerator.iterate() for details)
        async for result in job:

            # We'll only collect text here, but the result could contain other updates
            full_completion += result.get("text", "")

            # Output marker to console to confirm tasks running asynchronously
            print(marker, end = ""); sys.stdout.flush()

            # Cancel the second job after 300 characters, to make the control flow a little less trivial.
            # Normally, the last iteration of a job will free the cache pages allocated to it, so if we're ending
            # the job prematurely we have to manually call job.cancel() here
            if marker == "1" and len(full_completion) > 300:
                await job.cancel()
                break

        return full_completion

    tasks = [run_job(prompt, str(i)) for i, prompt in enumerate(prompts)]
    outputs = await asyncio.gather(*tasks)

    print()
    print()
    for i, output in enumerate(outputs):
        print(f"Output {i}")
        print("-----------")
        print(output)
        print()

    await generator.close()

if __name__ == "__main__":
    asyncio.run(main())



