from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
from voli.end_to_end_qa import EndToEndQA
import dpr
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("./results_hcm/checkpoint-6000/")
tokenizer = AutoTokenizer.from_pretrained("./results_hcm/checkpoint-6000/")

DATASET = './data/hcm'
OUR_MODEL = False

print("Loading Pickle Files")
database = pickle.load(open(DATASET + '_database.pkl', 'rb'))
questions_as_tfidf = pickle.load(open(DATASET + '_questions_as_tfidf.pkl', 'rb'))
questions_as_str = pickle.load(open(DATASET + '_questions_as_str.pkl', 'rb'))
tfidf_vocab = pickle.load(open(DATASET + '_tfidf_vocab.pkl', 'rb'))
tfidf_df = pickle.load(open(DATASET + '_tfidf_df.pkl', 'rb'))

if OUR_MODEL:
    print("Loading Model")
    model = EndToEndQA(database, questions_as_tfidf, questions_as_str, tfidf_vocab, tfidf_df)
    loaded_model = torch.load('./models/meqsum_coeff_0.5_endtoendqa_best_loss=33.78.pt')
    model.load_state_dict(loaded_model['state_dict'])
    #model.cuda()
    model.eval()

dpr_instance = dpr.DPR(database, questions_as_str, dataset=DATASET)

def chatbot_response(msg):
    # DPR
    dpr_answer = dpr_instance.dpr(msg)

    # GAR
    input_ids = tokenizer(msg, return_tensors="pt", truncation=True, padding=True).input_ids
    generated = model.generate(input_ids)[0]
    output = tokenizer.decode(generated, skip_special_tokens=True)
    gar_answer = dpr_instance.dpr(msg + ' ' + output)

    if OUR_MODEL:
        # Own model
        model_outputs = model(msg.strip())
        print('Generated FAQ:', model_outputs[6])
        print('Matched FAQ:', model_outputs[5])
        selected_answers = ' '.join(model_outputs[2])
        print('Selected Answers:', selected_answers, '\n')
        del model_outputs[4]
        del model_outputs[3]
        del model_outputs[1]
        del model_outputs[0]
        del model_outputs
        torch.cuda.empty_cache()

    responses = {
        "DPR": dpr_answer,
        "GAR": gar_answer,
    }

    str_responses = ' <br> '.join(["System " + str(idx+1) + ':\n' + item[1] for idx, item in enumerate(responses.items())])
    return str_responses