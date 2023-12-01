
import fire

from llama_recipes.inference.safety_check import safety_check

def main(
    prompt: str,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):

        result = safety_check(prompt=prompt, 
                    ckpt_dir=ckpt_dir,
                    tokenizer_path=tokenizer_path,
                    max_seq_len=max_seq_len,
                    max_batch_size=max_batch_size,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,)
        
        print(result)
        # is_safe = result[0].split(" ")[0] == "safe"    

        # print(f"Is safe? {is_safe}")
            
        # report = result[0]

        # print(f"Report: {report}")


if __name__ == "__main__":
    fire.Fire(main)