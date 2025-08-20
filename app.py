import os
import time
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import datetime
from io import StringIO
import base64

# Load env vars
load_dotenv()

# Module selector
st.set_page_config(page_title="Job Application Manager", layout="wide")
module = st.sidebar.radio("🧭 Select Module", ["📄 Cover Letter Finder (M1)", "📬 Email Status Tracker (M3)"])

# ──────────────────────────────────────────────
# MODULE 1: Cover Letter RAG
# ──────────────────────────────────────────────
if module == "📄 Cover Letter Finder (M1)":
    from modules.m1_cover_letter_rag.loader import load_cover_letters
    from modules.m1_cover_letter_rag.vector import (
        load_and_split_docs,
        create_vectorstore,
        load_vectorstore,
        build_ensemble_retriever
    )
    from modules.m1_cover_letter_rag.ranker import rerank_documents
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.combine_documents import create_stuff_documents_chain

    st.title("📄 Job Application Copilot (Cover Letter RAG + Groq)")

    # Sidebar for M1
    with st.sidebar:
        st.header("💾 Embedding Setup")
        folder = st.text_input("📁 Cover Letter Folder", value="data/cover_letters")
        embed_btn = st.button("Build Vectorstore")
        show_sources = st.checkbox("🔍 Show Sources", value=True)

    # LLM and prompt setup
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="LLaMA3-8b-8192")

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the context below. Be precise and concise.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Embedding
    if embed_btn:
        with st.spinner("Loading and embedding documents..."):
            docs = load_and_split_docs(folder)
            vectorstore = create_vectorstore(docs)
            st.session_state.docs = docs
            st.session_state.vectorstore = vectorstore
            st.success("✅ Vectorstore created and stored!")

    # Load vectorstore if not already
    if "vectorstore" not in st.session_state:
        vectorstore = load_vectorstore()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.docs = load_cover_letters(folder)

    # Query input
    query = st.text_input("🔎 What are you looking for?", placeholder="e.g., Cover letter for BMW Data Analyst")
    if query and "vectorstore" in st.session_state:
        with st.spinner("Retrieving documents..."):
            retriever = build_ensemble_retriever(st.session_state.docs, st.session_state.vectorstore)
            retrieved_docs = retriever.get_relevant_documents(query)

        with st.spinner("Reranking top documents..."):
            reranked_docs = rerank_documents(query, retrieved_docs, top_n=3)

        with st.spinner("Generating final answer with LLaMA3..."):
            start = time.process_time()
            response = document_chain.invoke({
                "context": reranked_docs,
                "input": query
            })
            runtime = time.process_time() - start

        print("🧪 Raw LLM response:", response)
        final_answer = response if isinstance(response, str) else response.get('output_text', str(response))

        st.subheader("🧠 Final Answer (Groq - LLaMA3)")
        st.write(final_answer)
        st.caption(f"⚙️ Generated in {runtime:.2f} seconds")

        # Comparison display
        st.subheader("🔬 Comparison: Before vs After Reranking")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟡 Before Reranking")
            for i, doc in enumerate(retrieved_docs[:3]):
                st.markdown(f"**Chunk {i+1}:** `{os.path.basename(doc.metadata.get('source', ''))}`")
                st.write(doc.page_content[:500] + "...")
                st.caption(f"Source: {doc.metadata.get('source')}")

        with col2:
            st.markdown("### 🟢 After Reranking")
            for i, doc in enumerate(reranked_docs):
                st.markdown(f"**Chunk {i+1}:** `{os.path.basename(doc.metadata.get('source', ''))}`")
                st.write(doc.page_content[:500] + "...")
                st.caption(f"Rerank Score: {doc.metadata.get('rerank_score', 'N/A')}, Source: {doc.metadata.get('source')}")

# ──────────────────────────────────────────────
# MODULE 3: Email Status Tracker (Placeholder)
# ──────────────────────────────────────────────
elif module == "📬 Email Status Tracker (M3)":
    st.title("📬 Email Status Tracker (Gmail + LLM)")

    # ---------- Sidebar Upload ----------
    st.sidebar.header("📂 Upload Application Log (.csv or .xlsx)")
    uploaded_file = st.sidebar.file_uploader("Drag and drop file here", type=["csv", "xlsx"])

    # ---------- Read File ----------
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl', header=0)

            # Drop fully empty columns (sometimes unnamed)
            df.dropna(axis=1, how='all', inplace=True)

            # Fix column names if any are unnamed or weird
            df.columns = [str(c).strip() for c in df.columns]

            # Fix date column: try to parse likely date columns
            for col in df.columns:
                if "date" in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

            # 🧠 Ensure 'Status' column exists
            if "Status" not in df.columns:
                df["Status"] = None        
            
            st.session_state.applications = df
            st.success("✅ Application file loaded successfully!")

        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")

    # ---------- Manual Entry ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Manual Entry")
    with st.sidebar.form("manual_entry_form"):
        company = st.text_input("Company")
        job_title = st.text_input("Job Title")
        app_date = st.date_input("Application Date", value=datetime.date.today())
        submitted = st.form_submit_button("➕ Add to Tracker")
        if submitted and company and job_title:
            new_entry = {
                "Company": company,
                "Job Title": job_title,
                "Application Date": pd.to_datetime(app_date)
            }
            if "applications" not in st.session_state:
                st.session_state.applications = pd.DataFrame(columns=new_entry.keys())
            st.session_state.applications = pd.concat(
                [st.session_state.applications, pd.DataFrame([new_entry])],
                ignore_index=True
            )
            st.success(f"✅ Added {company} - {job_title} to tracker!")

    # ---------- Display Table ----------
    st.subheader("📋 Your Job Applications")
    if "applications" in st.session_state and not st.session_state.applications.empty:
        if st.button("📨 Search Gmail for Application Responses"):
            with st.spinner("🔐 Authenticating with Gmail..."):
                from modules.m3_email_status_tracker.gmail_reader import authenticate_gmail, search_emails
                service = authenticate_gmail()
            
            results = []
            with st.spinner("🔍 Searching Gmail for each application..."):
                for i, row in st.session_state.applications.iterrows():
                    company = row.get("Company", "")
                    job_title = row.get("Job Title", "")
                    app_date = row.get("Application Date", pd.to_datetime("2000-01-01"))

                    # Construct Gmail search query
                    query = f"{company} {job_title}"
                    emails = search_emails(service, query, app_date)

                    from modules.m3_email_status_tracker.gmail_reader import classify_email_status

                    # Prepare snippets and classify status
                    if emails:
                        snippets = "\n\n".join([f"- {email['subject']} | {email['snippet']}" for email in emails])
                        # Choose first relevant email (or loop if needed)
                        primary_email = emails[0]
                        status = classify_email_status(primary_email["subject"], primary_email["snippet"])
                    else:
                        snippets = "No emails found"
                        status = "No Response"

                    results.append({
                        "Email Matches": snippets,
                        "Status": status
                    })

            # Add results as new column
            result_df = pd.DataFrame(results)
            for col in ["Status", "Email Matches"]:
                if col in st.session_state.applications.columns:
                    st.session_state.applications.drop(columns=col, inplace=True)

            # Merge updated results
            st.session_state.applications = pd.concat(
                [st.session_state.applications.reset_index(drop=True), result_df.reset_index(drop=True)],
                axis=1
            )
            st.success("✅ Gmail search completed!")
            st.subheader("📬 Applications with Status")
            # Desired display columns
            desired_columns = ["Company", "Position", "Date of Application", "Status", "Email Matches"]

            # Only include the ones that actually exist
            available_columns = [col for col in desired_columns if col in st.session_state.applications.columns]

            if available_columns:
                st.dataframe(st.session_state.applications[available_columns])
            else:
                st.warning("⚠️ No recognizable columns found to display.")