import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import chromadb
from fastapi.middleware.cors import CORSMiddleware
from schemas import QueryRequest, QueryResponse
from dotenv import load_dotenv

origins = [
    "http://localhost:5173",
    "http://localhost:8080",
    "https://springbackend-zei7.onrender.com",
    "https://teachassist-delta.vercel.app"
]

app = FastAPI(title="RAG API with gpt 4o mini")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv("minimal.env")
# Initialize clients (Ensure OPENAI_API_KEY is set in .env):
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to vector db:

try:
    clientdb = chromadb.CloudClient(
    api_key=os.getenv("CRHOMADB_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
    database='prod'
    )
    collection = clientdb.get_collection("arabic_books")
except Exception as e:
    print(f"Failed to connect to Chroma Cloud: {str(e)}")
    collection = None

# Pydantic schemas with default top_k so payload can omit it
class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=10, ge=1)

class QueryResponse(BaseModel):
    answer: str
    context: list[str]


def retrieve_context(query: str, top_k: int) -> list[str]:
    if not collection:
        return []
    try:
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = embedding_response.data[0].embedding

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        documents = results.get("documents", [[]])

        return documents[0] if documents else []
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return []


def generate_answer(question: str, contexts: list[str]) -> str:
    source_knowledge = "\n---\n".join(contexts)

    # 2. Updated system prompt to include source_knowledge
    system_prompt = (
        "أنت المستشار الرقمي للمعلم، نظام خبير ذكي ومبادرة تربوية تهدف إلى مساندة "
        "المعلمين والمعلمات في المدارس لحل مشكلاتهم اليومية (التربوية، السلوكية، والقانونية الإدارية) "
        "فوراً، وبما يتوافق تماماً مع رؤية ورسالة وقوانين وزارة التربية والتعليم الرسمية المرفقة في ملفات المعرفة لديك.\n\n"

        f"السياق المتاح:\n{source_knowledge}\n\n"

        "قواعد الإجابة:\n"
        "1. أجب باللغة العربية الواضحة والمهنية.\n"
        "2. استخدم Markdown القياسي فقط لتنسيق الإجابة، ولا تستخدم HTML.\n"
        "3. ابدأ الإجابة بعنوان Markdown قصير باستخدام # أو ## عند الحاجة.\n"
        "4. بعد العنوان، أضف ملخصاً موجزاً من جملة أو جملتين يوضح الإجابة.\n"
        "5. استخدم القوائم النقطية (-) أو (*) عند عرض عدة نقاط.\n"
        "6. استخدم القوائم المرقمة فقط عندما يكون ترتيب الخطوات مهماً.\n"
        "7. استخدم **النص العريض** لتوضيح المصطلحات أو النقاط المهمة.\n"
        "8. استخدم `inline code` فقط عند الحاجة لعرض أسماء حقول أو أكواد تقنية.\n"
        "9. استخدم الجداول Markdown فقط عندما تكون المقارنة أو البيانات الجدولية مفيدة.\n"
        "10. لا تضع كل الإجابة داخل كتلة كود ```.\n"
        "11. لا تستخدم HTML مثل <div> أو <p> أو <br> أو inline styles.\n"
        "12. لا تضف Markdown غير ضروري؛ اجعل التنسيق واضحاً ونظيفاً وسهل القراءة.\n\n"

        "قواعد المحتوى:\n"
        "إذا كانت المعلومات المتاحة غير كافية للإجابة، فاعتذر للمستخدم وأوضح أن البيانات المتاحة "
        "غير كافية للإجابة بدقة، ولا تخترع أو تفترض معلومات غير موجودة.\n"
        "إذا كان السؤال لا يتعلق بالتعليم أو علم النفس أو التربية، فاعتذر للمستخدم وأوضح أن "
        "المحادثة مخصصة لهذه المجالات فقط.\n"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1
        )

        # 3. Fixed response indexing [0]
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        # 4. Return string fallback instead of None to keep Pydantic happy
        return "حدث خطأ أثناء توليد الإجابة. يرجى المحاولة لاحقاً."


@app.post("/ai/v1/query", response_model=QueryResponse)
async def rag_endpoint(payload: QueryRequest):
    contexts = retrieve_context(payload.question, payload.top_k)

    if contexts == []:
        return QueryResponse(answer="غير متواجد في قاعدة البيانات. يرجى المحاولة بسؤال آخر" , context=[])
    answer = generate_answer(payload.question, contexts)
    
    return QueryResponse(answer=answer, context=contexts)