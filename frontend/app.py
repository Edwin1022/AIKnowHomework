import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="LLM Chat App", layout="wide")

# --- Session State Management ---
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None

if "uploader_key_counter" not in st.session_state:
    st.session_state.uploader_key_counter = 0

# --- API Client Layer ---
def create_conversation():
    try:
        res = requests.post(f"{API_BASE_URL}/conversations")
        res.raise_for_status()
        conv = res.json()
        st.session_state.current_conv_id = conv["id"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create conversation: {e}")

def list_conversations():
    try:
        res = requests.get(f"{API_BASE_URL}/conversations")
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to list conversations: {e}")
        return []
    
def read_conversation(conv_id):
    try:
        res = requests.get(f"{API_BASE_URL}/conversations/{conv_id}")
        if res.status_code == 200:
            return res.json()
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to read conversation: {e}")
        return None

def update_conversation_title(conv_id, new_title):
    try:
        requests.patch(f"{API_BASE_URL}/conversations/{conv_id}", json={"title": new_title})
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to update conversation title: {e}")

def delete_conversation(conv_id):
    try:
        requests.delete(f"{API_BASE_URL}/conversations/{conv_id}")
        if st.session_state.current_conv_id == conv_id:
            st.session_state.current_conv_id = None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete conversation: {e}")

def send_chat_message(conv_id, prompt, uploaded_image=None):
    url = f"{API_BASE_URL}/conversations/{conv_id}/chat"
    data = {"content": prompt}
    files = None
    
    if uploaded_image:
        files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        
    try:
        with requests.post(url, data=data, files=files, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.exceptions.RequestException as e:
        yield f"\n\n**[Error connecting to backend: {e}]**"

# --- Sidebar: Conversation Management ---
with st.sidebar:
    st.title("💬 Chat History")
    
    if st.button("➕ New Conversation", use_container_width=True):
        create_conversation()
        st.rerun()
    
    st.divider()
    
    conversations = list_conversations()
    for conv in conversations:
        title = conv["title"] if conv["title"] else "New Conversation"
        
        col1, col2 = st.columns([8, 2])
        with col1:
            is_active = st.session_state.current_conv_id == conv["id"]
            button_type = "primary" if is_active else "secondary"
            
            if st.button(title, key=f"btn_{conv['id']}", type=button_type, use_container_width=True):
                st.session_state.current_conv_id = conv["id"]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                delete_conversation(conv["id"])
                st.rerun()

# --- Main Chat Area ---
if st.session_state.current_conv_id:
    conv_id = st.session_state.current_conv_id
    
    # Use helper to fetch details
    conv_data = read_conversation(conv_id)
    
    if conv_data:
        messages = conv_data.get("messages", [])
        
        title = conv_data["title"] if conv_data["title"] else "New Conversation"
        header_col1, header_col2 = st.columns([8, 2])
        with header_col1:
            st.header(title)
        with header_col2:
            with st.popover("✏️ Edit Title"):
                new_title = st.text_input("New Title", value=title)
                if st.button("Save Title"):
                    update_conversation_title(conv_id, new_title)
                    st.rerun()
        
        st.divider()
        
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        upload_container = st.container()
        
        dynamic_uploader_key = f"uploader_{conv_id}_{st.session_state.uploader_key_counter}"
        
        if prompt := st.chat_input("Type your message here..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            uploaded_image = st.session_state.get(dynamic_uploader_key)
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in send_chat_message(conv_id, prompt, uploaded_image):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                st.session_state.uploader_key_counter += 1
                st.rerun()
                    
        with upload_container:
             uploaded_file = st.file_uploader(
                "Attach an image (Bonus Requirement)", 
                type=["png", "jpg", "jpeg"], 
                key=dynamic_uploader_key 
            )
             
             if uploaded_file is not None:
                 st.image(uploaded_file, caption="Image ready to send", width=250)
            
    else:
        st.error("Conversation not found. It may have been deleted.")
        st.session_state.current_conv_id = None
        
else:
    st.title("🤖 Welcome to the LLM Chat")
    st.write("Select an existing conversation from the sidebar or click **New Conversation** to start.")