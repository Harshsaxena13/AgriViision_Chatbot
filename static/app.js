const state = {
  conversationId: window.localStorage.getItem("agroai_conversation_id"),
};

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const messageEl = document.querySelector("#message");
const diagnosisEl = document.querySelector("#diagnosis");
const diagnoseButton = document.querySelector("#diagnoseButton");

function fieldValue(id) {
  return document.querySelector(`#${id}`).value.trim() || null;
}

function addMessage(role, content) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = content;
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(event) {
  event.preventDefault();
  const text = messageEl.value.trim();
  if (!text) return;

  addMessage("user", text);
  messageEl.value = "";

  const submitButton = formEl.querySelector("button");
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/v1/assistant/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        conversation_id: state.conversationId,
        farmer_name: fieldValue("farmerName"),
        crop: fieldValue("crop"),
        disease: fieldValue("disease"),
        location: fieldValue("location"),
        language: "en",
      }),
    });

    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Assistant request failed.");
    }

    state.conversationId = body.conversation_id;
    window.localStorage.setItem("agroai_conversation_id", state.conversationId);
    addMessage("assistant", body.answer);
  } catch (error) {
    addMessage("error", error.message);
  } finally {
    submitButton.disabled = false;
    messageEl.focus();
  }
}

async function diagnoseImage() {
  const imageInput = document.querySelector("#image");
  const file = imageInput.files[0];
  if (!file) {
    diagnosisEl.textContent = "Select a crop image first.";
    return;
  }

  diagnoseButton.disabled = true;
  diagnosisEl.textContent = "Diagnosing image...";

  try {
    const formData = new FormData();
    formData.append("image", file);
    const response = await fetch("/api/v1/assistant/diagnose", {
      method: "POST",
      body: formData,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Image diagnosis failed.");
    }

    document.querySelector("#disease").value = body.disease;
    const confidence =
      body.confidence === null || body.confidence === undefined
        ? ""
        : ` Confidence: ${(body.confidence * 100).toFixed(1)}%.`;
    diagnosisEl.textContent = `Detected: ${body.disease}.${confidence}`;
  } catch (error) {
    diagnosisEl.textContent = error.message;
  } finally {
    diagnoseButton.disabled = false;
  }
}

formEl.addEventListener("submit", sendMessage);
diagnoseButton.addEventListener("click", diagnoseImage);

addMessage(
  "assistant",
  "Tell me what you see in the crop, or upload a wheat leaf image for disease detection."
);
