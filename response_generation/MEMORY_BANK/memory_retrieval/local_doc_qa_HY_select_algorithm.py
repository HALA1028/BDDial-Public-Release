from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
# from models.chatglm_llm import ChatGLM
import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter, NLTKTextSplitter
from memory_retrieval.configs.model_config import *
import datetime
from memory_retrieval.textsplitter import ChineseTextSplitter
from typing import List, Tuple, Union
from langchain.docstore.document import Document
import numpy as np
import json
from os import PathLike
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from sentence_transformers import SentenceTransformer
# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 2
RESPONSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class JsonMemoryLoader(UnstructuredFileLoader):
    def __init__(self, filepath,language,mode="elements"):
        super().__init__(filepath, mode=mode)
        self.filepath = filepath
        self.language = language
    def _get_metadata(self, date: str) -> dict:
        return {"source": date}
    
    def load(self,name):
        user_memories = []
        print(self.file_path)
        f = open(self.filepath, "r", encoding="utf-8")
        memories = json.loads(f.read())
        for user_name, user_memory in memories.items():
            if user_name != name:
                continue
        # print(user_memory)
            user_memories = []
            if 'history' not in user_memory.keys():
                continue
            for date, content in user_memory['history'].items():
                metadata = self._get_metadata(date)
                memory_str = f'时间{date}的对话内容：' if self.language=='cn' else f'Conversation content on {date}:'
                user_kw = '[|用户|]：' if self.language=='cn' else '[|User|]:'
                ai_kw = '[|AI助手|]：' if self.language=='cn' else '[|AI|]:'
                for i,(dialog) in enumerate(content):
                    query, response = dialog['query'],dialog['response']
                    # memory_str += f'Memory: '
                    tmp_str = memory_str
                    tmp_str += f'{user_kw} {query.strip()}; '
                    tmp_str += f'{ai_kw} {response.strip()}'
                    user_memories.append(Document(page_content=tmp_str, metadata=metadata)) 
                # memory_str += '\n'
                if 'summary' in user_memory.keys(): 
                    if date in user_memory['summary'].keys():
                        summary = f'时间{date}的对话总结为：{user_memory["summary"][date]}' if self.language=='cn' else f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                        # memory_str += summary
                        user_memories.append(Document(page_content=summary, metadata=metadata)) 
                # if 'personality' in user_memory.keys():
                #     if date in user_memory['personality'].keys():
                #         memory_str += f'日期{date}的对话分析为：{user_memory["personality"][date]}'
                # print(memory_str)
                # user_memories.append(Document(page_content=memory_str, metadata=metadata)) 
        f.close() 
        return user_memories
     
    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None,name=''
    ) -> List[Document]:
        """Load documents and split into chunks."""
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.load(name)
        results = _text_splitter.split_documents(docs)

        # for result in results:
        #     date,content = result.metadata['source'],result.page_content
        #     result.page_content = f'{content}'
        # print(docs[0])
        # print(results[0])
        # exit()
        return results

def load_file(filepath,language='cn'):
    if filepath.endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        if language=='cn':
            textsplitter = ChineseTextSplitter(pdf=True)
        else:
            textsplitter=RecursiveCharacterTextSplitter(pdf=True,separators=["\n\n", "\n", " ", "","Round"])
        # textsplitter = NLTKTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        # textsplitter = ChineseTextSplitter(pdf=False)
        textsplitter=RecursiveCharacterTextSplitter(pdf=True,separators=["Memory:"])
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs

def load_memory_file(filepath,user_name,language='cn'):
    loader = JsonMemoryLoader(filepath,language)
    docs = loader.load(user_name)
    # if language=='cn':
    # textsplitter = ChineseTextSplitter(pdf=False)
    # else:
    #     textsplitter = RecursiveCharacterTextSplitter()
    # textsplitter = ChineseTextSplitter(pdf=False)
    # docs = loader.load_and_split(textsplitter,user_name)
    return docs
 
def load_HY_file(filepath)-> List[Document]:
    user_memories = []

    with open(filepath, "r", encoding="utf-8") as f:
        memories = json.load(f)  # 加载整个 JSON 文件

    for key, entry in memories.items():
        if "Description" in entry:  # 确保 JSON 结构正确
            metadata = {"source": key}
            user_memories.append(Document(page_content=entry["Description"], metadata=metadata))

    return user_memories

def get_docs_with_score(docs_with_score):
    docs=[]
    Memory_item = False
    similarity_threshold = 700
    for doc, score in docs_with_score:
        if score <= similarity_threshold:  # 只保留相似度分数低于阈值的文档
            doc.metadata["score"] = score
            docs.append(doc)
            Memory_item = True
    return docs,Memory_item

def reset_selected_questions():
    """重置 selected_questions，全局清空已选问题记录"""
    global selected_questions
    selected_questions = set()

def get_HY_with_score(docs_with_score,selected_questions):

    docs=[]
    HY_item = None
    similarity_threshold = 800
    penalty_factor = 3

    for doc, score in docs_with_score:
        source_id = doc.metadata["source"]

        if source_id in selected_questions:
            score *= penalty_factor

        if score <= similarity_threshold:  # 只保留相似度分数低于阈值的文档
            doc.metadata["score"] = score
            docs.append(doc)
            HY_item=doc.metadata["source"]

    return docs,HY_item

def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i-1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, len(docs)-i)):
                for l in [i+k, i-k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        # print(doc0.metadata)
                        # exit()
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break
                        # print(doc0)
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
        id_list = sorted(list(id_set))
        id_lists = seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs

class LocalMemoryRetrieval:
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL_CN,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 language='cn'
                 ):
        self.language = language
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                #model_kwargs={'device': 'cuda'}
                                                )
        self.top_k = top_k
    
    def init_memory_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    user_name: str = None,
                                    cur_date: str = None):
        loaded_files = []
        # filepath = filepath.replace('user',user_name)
        # vs_path = vs_path.replace('user',user_name)
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None, None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                # try:
                docs = load_memory_file(filepath,user_name,self.language)
                print(f"{file} 已成功加载")
                loaded_files.append(filepath)
                # except Exception as e:
                #     print(e)
                #     print(f"{file} 未能成功加载")
                #     return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    # if user_name not in fullfilepath:
                    #     continue
                    try:
                        docs += load_memory_file(fullfilepath,user_name,self.language)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_memory_file(file,user_name,self.language)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                print(f'Load from previous memory index {vs_path}.')
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    #以下为新增HY内容######################################################################################################################
    def init_HY_vector_store(self,
                                    filepath: Union[str, List[str]],
                                    vs_path: Union[str, PathLike] = None,       
                                    ):
        loaded_files = []
        # filepath = filepath.replace('user',user_name)
        # vs_path = vs_path.replace('user',user_name)
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None, None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                # try:
                docs = load_HY_file(filepath)
                print(f"{file} 已成功加载")
                loaded_files.append(filepath)
                # except Exception as e:
                #     print(e)
                #     print(f"{file} 未能成功加载")
                #     return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    # if user_name not in fullfilepath:
                    #     continue
                    try:
                        docs += load_HY_file(fullfilepath)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_HY_file(file)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                print(f'Load from previous memory index {vs_path}.')
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files
    #######################################################################################################################################

    def load_memory_index(self,vs_path):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size=self.chunk_size
        return vector_store
    
    def search_memory(self,query,vector_store):

        # vector_store = FAISS.load_local(vs_path, self.embeddings)
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        # vector_store.chunk_size=self.chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=1)

        related_docs,Memory_item = get_docs_with_score(related_docs_with_score)
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
        pre_date = ''
        date_docs = []
        dates = []
        for doc in related_docs:
            doc.page_content = doc.page_content.replace(f'时间{doc.metadata["source"]}的对话内容：','').strip()
            if doc.metadata["source"] != pre_date:
                # date_docs.append(f'在时间{doc.metadata["source"]}的回忆内容是：{doc.page_content}')
                date_docs.append(doc.page_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}' 
        # memory_contents = [doc.page_content for doc in related_docs]
        # memory_contents = [f'在时间'+doc.metadata['source']+'的回忆内容是：'+doc.page_content for doc in related_docs]
        print(date_docs)
        return date_docs, ', '.join(dates) ,Memory_item
    
    def search_HY(self,query,vector_store,selected_questions):

        # vector_store = FAISS.load_local(vs_path, self.embeddings)
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        # vector_store.chunk_size=self.chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=1)

        related_docs,HY_item = get_HY_with_score(related_docs_with_score,selected_questions)
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)

        if not related_docs:
            print("没有相关HY")
            return [], "", None, None  

        print(related_docs)
        
        page_content = related_docs[0].page_content
        
        with open(os.path.join(RESPONSE_ROOT, 'memories', 'HY_origin.json'), 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        matched_item = None
        for key, value in data.items():
            if value["Description"] == page_content:
                matched_item = value["Item"]

        pre_date = ''
        date_docs = []
        dates = []

        for doc in related_docs:
            doc.page_content = doc.page_content.replace(f'时间{doc.metadata["source"]}的对话内容：','').strip()
            if doc.metadata["source"] != pre_date:
                # date_docs.append(f'在时间{doc.metadata["source"]}的回忆内容是：{doc.page_content}')
                date_docs.append(doc.page_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}' 

        return date_docs, ', '.join(dates) ,HY_item,matched_item

# 加载 Sentence-BERT 中文模型（可以换成更适合的模型）
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def compute_sten_sbert_similarity(U, text):
    """ 使用 Sentence-BERT 计算文本相似度 """
    # embeddings = sbert_model.encode([U, text])
    # similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    similarity = cosine_similarity([U], [text])[0][0]
    return similarity

def compute_sten_tfidf_similarity(U, text):
    """ 使用 TF-IDF 计算文本相似度，支持中文分词 """
    U_cut = " ".join(jieba.cut(U))
    text_cut = " ".join(jieba.cut(text))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([U_cut, text_cut])  # 第一项是用户输入，第二项是文本项
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])  # 计算 U 和文本的相似度
    return similarity[0][0] 

def Sim_sent(U, i):
    return compute_sten_sbert_similarity(U, i)  # 句子级相似度

def compute_word_sbert_similarity(sentence, keywords_emb,keywords):
    """ 计算输入句子和多个关键词的平均相似度 """
    # if not keywords:
    #     return {}

    # # 获取所有文本的 SBERT 向量
    # embeddings = sbert_model.encode([sentence] + keywords)
    
    # # 计算 sentence (第一个向量) 与所有关键词向量的相似度
    # similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
    
    # return {kw: float(sim) for kw, sim in zip(keywords, similarities)}
    # if keywords is None or len(keywords) == 0:
    #     return 0
    if not keywords:
         return {}
    similarities = cosine_similarity([sentence], keywords_emb[0:])[0]  # 计算 U 与所有关键词的相似度
    return {kw: float(sim) for kw, sim in zip(keywords, similarities)}

def compute_word_tfidf_similarity(sentence, keywords):
    # 将所有关键词和句子组成语料库
    corpus = [sentence] + keywords
    
    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 计算相似度
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # 返回结果
    return {kw: float(sim) for kw, sim in zip(keywords, similarities[0])}

def Sim_key(U, i ,a):
    return compute_word_sbert_similarity(U, i, a)  # 关键词级相似度（这里可以拓展，比如只提取关键词）

def compute_relatedness_matrix(I):
    """ 计算 I 中所有问题之间的相关性 """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(I)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix 

def adaptive_hy_selection(U, I, t, count, similarity_weights=0.7, decay=0.2, threshold=0.5, delta=0.9):

    while True:
        U_emb = sbert_model.encode(U) 
        scores = {}
        scores_sent = {}
        scores_kw = {}

        # Step 1: Compute similarity scores with decay factor
        for key, value in I.items():
            S_sent = Sim_sent(U_emb, value["Description_emb"])  # 对比 I 矩阵的 Description 部分
            S_key = Sim_key(U_emb, value['Key_word_emb'],value['Key_word'])
            S_key_total = sum(S_key.values())
            S_key_avg = S_key_total / len(S_key) if S_key else 0
            base_score = value.get("base_score", 0.5)
            S_i = (similarity_weights * S_sent + (1 - similarity_weights) * S_key_avg) * base_score
            S_i *= decay ** count.get(key, 0)
            scores[key] = S_i
            scores_sent[key] = S_sent
            scores_kw[key] = S_key_avg
        
        # Step 2: Apply threshold filtering
        tau_t = max(threshold * (delta ** t), 0.3) 
        i_star = max(scores, key=scores.get)  # 选择最大得分的项目
        
        # Step 3: Check if the selected question exceeds the threshold
        if scores[i_star] >= tau_t:
        #    print(f"Selected item: {I[i_star]['Item']}")
            #count[i_star] = count.get(i_star, 0) + 1  # Step 4: Update question count
            
        #     # Step 5: Retrieve the most related question with decay
        #     related_scores = {}
            # if i_star in R:
            #     for rel in R[i_star]:
            #         j = rel["item"]
            #         related_scores[j] = rel["relevance"] * (decay ** count.get(j, 0))
            
        #    # 找到分数最高的相关问题（如果存在）
        #     most_related = max(related_scores, key=related_scores.get) if related_scores else None
            
        #     # 设定 0.5 分数门槛
        #     i_rel = most_related if (most_related and related_scores[most_related] >= 0.5) else ''
            
        #     if i_rel:
        #         print(f"Stored related item: {I[i_rel]['Item']}")
            
            return i_star, I[i_star]["Item"], I[i_star]["Description"]
    
        else:
            print("Continue general dialogue")
            return '', '', ''

def preprocess_I(I):
    """ 预计算 I 中的嵌入并缓存 """
    for key, value in I.items():
        value["Description_emb"] = sbert_model.encode(value['Description'], normalize_embeddings=True)
        if isinstance(value["Key_word"], list):  # 关键词可能是列表
            value["Key_word_emb"] = np.array([sbert_model.encode(kw, normalize_embeddings=True) for kw in value["Key_word"]])
        elif isinstance(value["Key_word"], str):  # 关键词是单个字符串
            value["Key_word_emb"] = np.array([sbert_model.encode(value["Key_word"], normalize_embeddings=True)])
        else:
            value["Key_word_emb"] = None
    return I

def main():
    # 加载 JSON 文件
    with open(os.path.join(RESPONSE_ROOT, 'memories', 'HY_inf.json'), 'r', encoding='utf-8') as f:
        I = json.load(f)

    I = preprocess_I(I)
    with open(os.path.join(RESPONSE_ROOT, 'evaluation', 'Relevance_Matrix.json'), 'r', encoding='utf-8') as file:
        R = json.load(file)
    
    count = {}  # 记录问题被问过的次数
    t=0
        
    while True:
        # 接收用户输入
        query = input("\nP: ")
        
        # 将用户输入作为 U 传递给 adaptive_hy_selection
        i_star,i_rel=adaptive_hy_selection(query, I, R, t, count )
        print("i_star:",i_star,"\n","i_rel:",i_rel)

        t+=1


if __name__ == "__main__":
    main()
