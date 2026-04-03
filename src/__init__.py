"""
EMMA — Emergency Medicine Mentoring Agent
DTI 5125, Group 23  (Jaxen Dutta · Acassia Arnaud · Yifei Yu)

Package layout
──────────────
emma.data           — loaders for MedQA, MedMCQA, textbooks
emma.vectorstore    — build & query the FAISS textbook index
emma.retrieval      — RAG pipeline (query → retrieve → prompt)
emma.knowledge_graph— AMG-RAG style MKG construction
emma.llm            — Ollama interface + model switching
emma.classify       — A1-style specialty classifier on MedQA questions
emma.cluster        — A2-style clustering on question embeddings
emma.quiz           — quiz mode logic and critique generation
emma.api            — FastAPI app (Dialogflow webhook backend)
"""
