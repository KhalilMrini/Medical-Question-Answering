from haystack.nodes import FARMReader
from haystack.document_stores import FAISSDocumentStore
import pickle
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline
import os

class DPR(object):

    def __init__(self, database, questions_as_str, dataset='meqsum') -> None:
        self.dataset = dataset
        self.database = database
        self.questions_as_str = questions_as_str
        self.faiss_name = dataset + "_faiss_document_store"
        
        existing_doc_store = self.faiss_name + ".db" in os.listdir('.')
        if existing_doc_store:
            document_store = FAISSDocumentStore.load(
                index_path=self.faiss_name + ".index", 
                config_path=self.faiss_name + ".config")
        else:
            document_store = FAISSDocumentStore(
                faiss_index_factory_str="Flat",
                sql_url="sqlite:///{}.db".format(self.faiss_name))
            dicts = [{"content": sent, 
                      "meta": {
                          "name": "Ques {0}".format(qidx+1), "idx": qidx}} 
                          for qidx, sent in enumerate(self.questions_as_str)]
            document_store.write_documents(dicts)
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

        retriever_dpr = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            max_seq_len_query=512,
            max_seq_len_passage=256,
            batch_size=8,
            use_gpu=True,
            embed_title=False,
            use_fast_tokenizers=True,
        )
        if not existing_doc_store:
            document_store.update_embeddings(retriever_dpr)
            document_store.save(index_path=self.faiss_name + ".index", 
                                config_path=self.faiss_name + ".config")
        self.pipe = ExtractiveQAPipeline(reader, retriever_dpr)

    def dpr(self, message):
        ## RETRIEVAL OF QUESTION

        prediction = self.pipe.run(
            query=message, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )
        filtered_questions = [(getattr(ans, 'meta')['idx'], getattr(ans, 'context')) 
                              for ans in prediction['answers']]
        matched_question = filtered_questions[0][1]
        matched_idx = int(filtered_questions[0][0])
        print("MATCHED QUESTION", matched_idx)
        print(matched_question)

        ## RETRIEVAL OF ANSWER

        ans_suffix = "_" + str(matched_idx)
        existing_doc_store_ans = self.faiss_name + ans_suffix + ".db" in os.listdir('.')
        if existing_doc_store_ans:
            document_store_ans = FAISSDocumentStore.load(
                index_path=self.faiss_name + ans_suffix + ".index", 
                config_path=self.faiss_name + ans_suffix + ".config")
        else:
            document_store_ans = FAISSDocumentStore(
                faiss_index_factory_str="Flat", 
                sql_url="sqlite:///{}_{}.db".format(self.faiss_name, str(matched_idx)))
            dicts_ans = [{"content": sent, 
                         "meta": {
                             "name": "Answer {0}".format(qidx+1), "idx": qidx}} 
                             for qidx, sent in enumerate(self.database[matched_idx])]
            document_store_ans.write_documents(dicts_ans)
        reader_ans = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

        retriever_dpr_ans = DensePassageRetriever(
            document_store=document_store_ans,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            max_seq_len_query=512,
            max_seq_len_passage=256,
            batch_size=8,
            use_gpu=True,
            embed_title=False,
            use_fast_tokenizers=True,
        )
        if not existing_doc_store_ans:
            document_store_ans.update_embeddings(retriever_dpr_ans)
            document_store_ans.save(
                index_path=self.faiss_name + ans_suffix + ".index", 
                config_path=self.faiss_name + ans_suffix + ".config")
        pipe_ans = ExtractiveQAPipeline(reader_ans, retriever_dpr_ans)
        prediction_ans = pipe_ans.run(
            query=message, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )
        matched_answers = [(getattr(ans,'meta')['idx'], getattr(ans,'context')) 
                           for ans in prediction_ans['answers']][:3]
        sorted_answers = sorted(matched_answers, key=lambda x:x[0])
        print("SORTED ANSWERS")
        print(sorted_answers)
        sorted_answers = [answer[1] for answer in sorted_answers]
        return '\n'.join(sorted_answers)
