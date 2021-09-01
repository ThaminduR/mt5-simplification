from transformers import MT5ForConditionalGeneration, T5Tokenizer
import argparse
import json
import torch
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictions')
    parser.add_argument('filepath', type=str, help='File containing the source sentences, one sentence per line.')
    parser.add_argument('--model-path', type=str, help=f'Model path to generate from.', required=True)
    parser.add_argument('--task', type=str, help=f'Task prefix.', required=True)
    parser.add_argument('--count', type=int, help=f'Number of predicted sentences. Defaults to no. of source sentences.')
    parser.add_argument('--max_length', type=int, help=f'Max length of predicted sentence. Defaults to 700.')
    parser.add_argument('--output', type=str, help=f'Output file name.', required=True)
    parser.add_argument('--verbosity', type=str,help=f'Specify verbosity using h (High), m (Medium), l (Low). Defaults to m.')
    parser.add_argument('--temp', type=int,help=f'Temperature for random sampling.')
    parser.add_argument('--topk', type=int,help=f'Top k for random sampling.')
    parser.add_argument('--topp', type=int,help=f'Top p for random sampling.')
    parser.add_argument('--rep_pen', type=int,help=f'Repetition penalty.')

    args = parser.parse_args()

    outfile = open(args.output,'w')
    source_file = open(args.filepath, 'r')
    source_sentences = source_file.readlines()
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting predictions for %s task." % args.task)
    logging.info("%d Source Sentences found." % len(source_sentences))
    logging.info("Predicting %d Sentences." % args.count)
    logging.info("Loading %s model." % args.model_path.split("/")[-1])
    logging.info("Starting predictions on %s." % 'cuda' if torch.cuda.is_available() else 'cpu. Unable to find cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer from path
    model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    logging.info("Model and Tokenizer loaded successfully. Starting predictions.")

    # Set max length
    max_length = args.max_length if (args.max_length) else 700
    logging.info("Max length set to %d." % max_length)

    # Set verbosity.
    verbose=500
    if args.verbosity=="h":
        verbose=100
    elif args.verbosity=="l":
        verbose=1000

    logging.info("Verbosity set to %s." % args.verbosity)

    # Set count
    count= args.count if(args.count) else len(source_sentences)

    temp = args.temp if(args.temp) else 1.0
    logging.info("Temperature set to %d." % temp)

    topk = args.topk if(args.topk) else 50
    logging.info("Top k set to %d." % topk)

    topp = args.topp if(args.topp) else 1.0
    logging.info("Top p set to %d." % topp)

    rep_pen = args.rep_pen if(args.rep_pen) else 1.0
    logging.info("Repeptition penalty set to %d." % rep_pen)

        
    i = 0
    for line in source_sentences[:count]:
        # Attach task prefix.
        line = args.task + ": " + line

        input_ids = tokenizer(line, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        output_ids = model.generate(input_ids=input_ids, do_sample=True,temperature=temp, max_length=max_length, top_k=topk, top_p=topp, repetition_penalty= rep_pen )
        out = tokenizer.decode(output_ids[0])

        # Remove pad and eos tokens.
        out = out.strip().replace('<pad>','').replace('</s>','').strip(" ")

        # Fix zero-width joiner issue.
        out = out.replace("\u0dca \u0dbb", "\u0dca\u200d\u0dbb").replace("\u0dca \u0dba", "\u0dca\u200d\u0dba")
        outfile.write('%s \n' % out)
        i += 1
        if(i%500==0): logging.info("%s sentences completed." % i)
