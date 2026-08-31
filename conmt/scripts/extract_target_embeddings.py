import argparse
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--dictionary")
    parser.add_argument("--output")

    args = parser.parse_args()

    model = torch.load(args.model_path)
    print(model.keys())

    
    target_embed = model["model"]["decoder.target_embed.weight"]

    print(target_embed.shape)

    # possibly save memory
    n_words = target_embed.shape[0]

    special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]

    dictionary = {v: i for i, v in enumerate(special_tokens)}

    with open(args.dictionary, 'r', encoding="utf-8") as d:
        lines = d.readlines()
        for line in lines:
            token, field = line.rstrip().rsplit(" ", 1)
            if token in special_tokens:
                continue
            dictionary[token] = len(dictionary)

    assert len(dictionary)==n_words
    
    # cdgm_textgen reuires this format as an input for pre-trained embeddings
    with open(f"{args.output}.txt", "w") as out:
        out.write("{} {}".format(target_embed.shape[0], target_embed.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in target_embed[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")
