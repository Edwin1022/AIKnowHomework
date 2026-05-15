import streamlit as st
import requests
import re
import pandas as pd
import altair as alt
from typing import Any, Generator, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="LLM Chat App", layout="wide")

DEMO_USERS = ["alice@example.com", "bob@example.com"]

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b"
]

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

# --- Session State Management ---
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None

if "current_user_email" not in st.session_state:
    st.session_state.current_user_email = DEMO_USERS[0]
    
if "current_model" not in st.session_state:
    st.session_state.current_model = AVAILABLE_MODELS[0]

if "uploader_key_counter" not in st.session_state:
    st.session_state.uploader_key_counter = 0

if "forking_msg_id" not in st.session_state:
    st.session_state.forking_msg_id = None

if "forking_content" not in st.session_state:
    st.session_state.forking_content = ""

# --- API Client Layer ---
def create_conversation() -> None:
    try:
        res = requests.post(
            f"{API_BASE_URL}/conversations",
            json={"user_email": st.session_state.current_user_email},
        )
        res.raise_for_status()
        conv = res.json()
        st.session_state.current_conv_id = conv["id"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create conversation: {e}")

def list_conversations() -> list[dict[str, Any]]:
    try:
        res = requests.get(
            f"{API_BASE_URL}/conversations",
            params={"user_email": st.session_state.current_user_email},
        )
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to list conversations: {e}")
        return []
    
def read_conversation(conv_id: str) -> Optional[dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/conversations/{conv_id}")
        if res.status_code == 200:
            return res.json()
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to read conversation: {e}")
        return None

def update_conversation_title(conv_id: str, new_title: str) -> None:
    try:
        requests.patch(f"{API_BASE_URL}/conversations/{conv_id}", json={"title": new_title})
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to update conversation title: {e}")

def delete_conversation(conv_id: str) -> None:
    try:
        requests.delete(f"{API_BASE_URL}/conversations/{conv_id}")
        if st.session_state.current_conv_id == conv_id:
            st.session_state.current_conv_id = None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete conversation: {e}")

def send_chat_message(
    conv_id: str, 
    prompt: str, 
    branch_id: int = 0,
    uploaded_image: Optional[UploadedFile] = None, 
    model_choice: str = AVAILABLE_MODELS[0]
) -> Generator[str, None, None]:
    url = f"{API_BASE_URL}/conversations/{conv_id}/chat"
    data: dict[str, str | int] = {"branch_id": branch_id, "content": prompt, "model_choice": model_choice}
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

def fork_message(
    conv_id: str, 
    message_id: str, 
    content: str,
    uploaded_image: Optional[UploadedFile] = None, 
    model_choice: str = AVAILABLE_MODELS[0]
):
    url = f"{API_BASE_URL}/conversations/{conv_id}/messages/{message_id}/fork"
    data: dict[str, str] = {"content": content, "model_choice": model_choice}
    files = None

    if uploaded_image:
        files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}

    try:
        with requests.post(url, data=data, files=files, stream=True) as r:
            r.raise_for_status()
            new_branch_id = int(r.headers.get("X-Branch-Id", 1))
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
            # store branch_id so caller can update session state after iteration
            st.session_state._last_fork_branch_id = new_branch_id
    except requests.exceptions.RequestException as e:
        yield f"\n\n**[Error connecting to backend: {e}]**" 
        
def get_user_cost(user_email: str) -> Optional[dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/users/{user_email}/cost")
        if res.status_code == 200:
            return res.json()
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch cost data: {e}")
        return None

# --- Cost Analytics Dashboard ---
@st.dialog("📊 Cost Analytics Dashboard", width="large")
def show_cost_dashboard():
    st.markdown(f"### 💰 Spend Dashboard")
    st.caption(f"User: `{st.session_state.current_user_email}`")
    
    cost_data = get_user_cost(st.session_state.current_user_email)
    
    if cost_data:
        st.metric(label="Total API Spend", value=f"${cost_data.get('total_cost_usd', 0):.4f}")
        
        breakdown = cost_data.get("breakdown", [])
        if breakdown:
            st.write("**Cost by Model**")
            
            df = pd.DataFrame(breakdown)
            
            chart = alt.Chart(df).mark_bar(color="red").encode(
                x=alt.X("cost_usd:Q", title="Cost ($)"),
                y=alt.Y(
                    "model:N", 
                    title=None, 
                    sort="-x", # Sorts the longest bars to the top
                    axis=alt.Axis(labelLimit=500) # Prevents Streamlit from cutting off long model names
                ),
                tooltip=["model", "cost_usd"]
            )
            st.altair_chart(chart, use_container_width=True)
            
            display_df = df.rename(columns={
                "model": "Model", 
                "prompt_tokens": "Input Tokens",
                "completion_tokens": "Output Tokens",
                "cost_usd": "Cost ($)"
            })
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("No tokens used yet. Start chatting!")

# --- Sidebar: Conversation Management ---
with st.sidebar:
    selected_user = st.selectbox(
        "👤 Current User",
        DEMO_USERS,
        index=DEMO_USERS.index(st.session_state.current_user_email),
    )
    if selected_user != st.session_state.current_user_email:
        st.session_state.current_user_email = selected_user
        st.session_state.current_conv_id = None
        st.rerun()
        
    st.session_state.current_model = st.selectbox(
        "🧠 Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.current_model)
    )
    
    if st.button("📊 View Cost Analytics", use_container_width=True):
        show_cost_dashboard()

    st.divider()

    if st.button("➕ New Conversation", use_container_width=True):
        create_conversation()
        st.rerun()
        
    st.title("💬 Chats")

    conversations = list_conversations()
    for conv in conversations:
        title = conv["title"] if conv["title"] else "New Conversation"
        
        col1, col2 = st.columns([8, 2])
        with col1:
            is_active = st.session_state.current_conv_id == conv["id"]
            button_type = "primary" if is_active else "secondary"
            
            if st.button(title, key=f"btn_{conv['id']}", type=button_type, use_container_width=True):
                st.session_state.current_conv_id = conv["id"]
                st.session_state.forking_msg_id = None
                st.session_state.forking_content = ""
                st.rerun()
        with col2:
            if st.button("", icon=":material/delete:", key=f"del_{conv['id']}", help="Delete conversation"):
                delete_conversation(conv["id"])
                st.rerun()

# --- Main Chat Area ---
if st.session_state.current_conv_id:
    conv_id = st.session_state.current_conv_id
    
    conv_data = read_conversation(conv_id)
    
    if conv_data:
        messages = conv_data.get("messages", [])
        
        title = conv_data["title"] if conv_data["title"] else "New Conversation"
        header_col1, header_col2 = st.columns([8, 2])
        with header_col1:
            st.header(title)
        with header_col2:
            with st.popover("Edit Conversation Title"):
                new_title = st.text_input("New Conversation Title", value=title)
                if st.button("Save Conversation Title"):
                    update_conversation_title(conv_id, new_title)
                    st.rerun()
        
        st.divider()

        # --- Build b0_turns: {b0_turn_number: [user_msg, asst_msg]} ---
        b0_turns: dict[int, list[dict[str, Any]]] = {}
        for m in messages:
            if m["branch_id"] == 0:
                b0_turns.setdefault(m["turn_number"], []).append(m)
        for t in b0_turns:
            b0_turns[t].sort(key=lambda m: m["sequence_number"])

        # --- Build fork_map: {b0_turn: {branch_id: {local_turn: [msgs]}}} ---
        fork_map: dict[int, dict[int, dict[int, list[Any]]]] = {}
        for m in messages:
            if m["branch_id"] == 0 or m["fork_start_seq"] is None:
                continue
            b0_turn = (m["fork_start_seq"] + 1) // 2
            (fork_map
                .setdefault(b0_turn, {})
                .setdefault(m["branch_id"], {})
                .setdefault(m["turn_number"], [])
                .append(m))
        for b0t in fork_map:
            for bid in fork_map[b0t]:
                for lt in fork_map[b0t][bid]:
                    fork_map[b0t][bid][lt].sort(key=lambda m: m["sequence_number"])

        # --- Derive current_branch for the chat input ---
        current_branch = 0
        for b0_turn in sorted(b0_turns):
            if b0_turn in fork_map:
                all_bids: list[int] = [0] + sorted(fork_map[b0_turn])
                idx = st.session_state.get(f"branch_idx_{conv_id}_{b0_turn}", 0)
                sel: int = all_bids[min(int(idx), len(all_bids) - 1)]
                if sel != 0:
                    current_branch = sel
                    break

        # --- Walk branch-0 turns ---
        fork_action: dict[str, Any] | None = None  # set when user submits an inline edit

        for b0_turn in sorted(b0_turns):
            pair_b0 = b0_turns[b0_turn]
            forks_here = fork_map.get(b0_turn, {})
            all_bids = [0] + sorted(forks_here)

            # Arrow toggle (only when multiple branches exist at this turn)
            if len(all_bids) > 1:
                idx_key = f"branch_idx_{conv_id}_{b0_turn}"
                idx = min(int(st.session_state.get(idx_key, 0)), len(all_bids) - 1)

                col_prev, col_label, col_next = st.columns([1, 8, 1])
                with col_prev:
                    if st.button("←", key=f"prev_{conv_id}_{b0_turn}", disabled=(idx == 0)):
                        st.session_state[idx_key] = idx - 1
                        st.rerun()
                with col_label:
                    st.caption(f"Version {idx + 1} / {len(all_bids)}")
                with col_next:
                    if st.button("→", key=f"next_{conv_id}_{b0_turn}", disabled=(idx == len(all_bids) - 1)):
                        st.session_state[idx_key] = idx + 1
                        st.rerun()

                selected_branch = int(all_bids[idx])
            else:
                selected_branch = 0

            if selected_branch == 0:
                user_msg = next(m for m in pair_b0 if m["role"] == "user")
                asst_msg = next((m for m in pair_b0 if m["role"] == "assistant"), None)

                is_editing = st.session_state.forking_msg_id == user_msg["id"]
                submit_clicked = False
                cancel_clicked = False
                edited_content = ""

                with st.chat_message("user"):
                    if is_editing:
                        edited_content = st.text_area(
                            "Edit message",
                            value=st.session_state.forking_content,
                            key=f"edit_area_{user_msg['id']}",
                            label_visibility="collapsed",
                        )
                        col_ok, col_cancel = st.columns([1, 1])
                        with col_ok:
                            submit_clicked = st.button("Submit", key=f"submit_{user_msg['id']}", type="primary")
                        with col_cancel:
                            cancel_clicked = st.button("Cancel", key=f"cancel_{user_msg['id']}")
                    else:
                        st.markdown(user_msg["content"].split("🖼️")[0].strip())
                        if st.button("Edit", key=f"edit_{user_msg['id']}"):
                            st.session_state.forking_msg_id = user_msg["id"]
                            raw_text = user_msg["content"].split("🖼️")[0]
                            clean_edit_text = re.sub(
                                r'!\[.*?\]\(data:.*?;base64,.*?\)', 
                                '', 
                                raw_text, 
                                flags=re.DOTALL
                            ).strip()
                            st.session_state.forking_content = clean_edit_text
                            st.rerun()

                if cancel_clicked:
                    st.session_state.forking_msg_id = None
                    st.session_state.forking_content = ""
                    st.rerun()

                if submit_clicked:
                    fork_action = {
                        "msg_id": user_msg["id"],
                        "content": edited_content,
                        "b0_turn": b0_turn,
                    }

                if asst_msg and not fork_action:
                    with st.chat_message("assistant"):
                        # MERGE FIX: Apply image memory split
                        st.markdown(asst_msg["content"].split("🖼️")[0].strip())

            else:
                # Show the selected fork's local turn 1 (the edited exchange)
                fork_local = forks_here[selected_branch]
                for lt_msg in fork_local.get(1, []):
                    with st.chat_message(lt_msg["role"]):
                        # MERGE FIX: Apply image memory split
                        st.markdown(lt_msg["content"].split("🖼️")[0].strip())

                # Show the fork's subsequent local turns (2, 3, …)
                for local_t in sorted(lt for lt in fork_local if lt > 1):
                    for lt_msg in fork_local[local_t]:
                        with st.chat_message(lt_msg["role"]):
                            # MERGE FIX: Apply image memory split
                            st.markdown(lt_msg["content"].split("🖼️")[0].strip())

                # Stop — do not render any more branch-0 turns
                break

        # Stream the forked response after the turn list (outside any chat_message context)
        if fork_action:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""
                for chunk in fork_message(conv_id, fork_action["msg_id"], fork_action["content"], model_choice=st.session_state.current_model):
                    full += chunk
                    display_text = full.split("🖼️")[0].strip()
                    placeholder.markdown(display_text + "▌")
                final_display = full.split("🖼️")[0].strip()
                placeholder.markdown(final_display)
                
            # Point the arrow at the new branch (it will be the last; clamp handles the index)
            st.session_state[f"branch_idx_{conv_id}_{fork_action['b0_turn']}"] = 999
            st.session_state.forking_msg_id = None
            st.session_state.forking_content = ""
            st.rerun()

        upload_container = st.container()

        dynamic_uploader_key = f"uploader_{conv_id}_{st.session_state.uploader_key_counter}"

        if prompt := st.chat_input("Type your message here..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            uploaded_image = st.session_state.get(dynamic_uploader_key)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in send_chat_message(conv_id, prompt, branch_id=current_branch, uploaded_image=uploaded_image, model_choice=st.session_state.current_model):
                    full_response += chunk
                    display_text = full_response.split("🖼️")[0].strip()
                    response_placeholder.markdown(display_text + "▌")
                
                final_display = full_response.split("🖼️")[0].strip()
                response_placeholder.markdown(final_display)
                
                st.session_state.uploader_key_counter += 1
                st.rerun()

        with upload_container:
            is_vision_model = st.session_state.current_model in VISION_MODELS
            
            uploader_label = "Attach an image" if is_vision_model else "⚠️ Image upload disabled (Switch to a Vision model - meta-llama/llama-4-scout-17b-16e-instruct)"
            
            uploaded_file = st.file_uploader(
                uploader_label, 
                type=["png", "jpg", "jpeg"], 
                key=dynamic_uploader_key,
                disabled=not is_vision_model
            )

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Image ready to send", width=250)
            
    else:
        st.error("Conversation not found. It may have been deleted.")
        st.session_state.current_conv_id = None
        
else:
    st.title("🤖 Welcome to the LLM Chat")
    st.write("Select an existing conversation from the sidebar or click **New Conversation** to start.")