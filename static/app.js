const VOICE_LANGUAGE_CODE = "en-IN";

const state = {
  conversationId: window.localStorage.getItem("agroai_conversation_id"),
  activeCapture: null,
  currentAudio: null,
  currentPlaybackResolve: null,
  isBusy: false,
  isDictating: false,
  isVoiceMode: false,
};

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const messageEl = document.querySelector("#message");
const diagnosisEl = document.querySelector("#diagnosis");
const diagnoseButton = document.querySelector("#diagnoseButton");
const imageInput = document.querySelector("#image");
const attachButton = document.querySelector("#attachButton");
const sendButton = document.querySelector("#sendButton");
const micButton = document.querySelector("#micButton");
const voiceModeButton = document.querySelector("#voiceModeButton");
const voiceStatusEl = document.querySelector("#voiceStatus");

const BrowserSpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition || null;
const canRecognizeSpeech = Boolean(BrowserSpeechRecognition);
const canSpeak = Boolean(window.speechSynthesis);

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

function setVoiceStatus(message) {
  voiceStatusEl.textContent = message;
}

function autoResizeInput() {
  messageEl.style.height = "auto";
  messageEl.style.height = `${Math.min(messageEl.scrollHeight, 160)}px`;
}

function setBusy(isBusy) {
  state.isBusy = isBusy;
  sendButton.disabled = isBusy;
  micButton.disabled = isBusy || state.isVoiceMode || !canRecognizeSpeech;
  voiceModeButton.disabled = (!state.isVoiceMode && isBusy) || !canRecognizeSpeech;
}

function setDictating(isDictating) {
  state.isDictating = isDictating;
  micButton.classList.toggle("active", isDictating);
  micButton.setAttribute("aria-pressed", String(isDictating));
}

function setVoiceMode(isVoiceMode) {
  state.isVoiceMode = isVoiceMode;
  voiceModeButton.classList.toggle("active", isVoiceMode);
  voiceModeButton.setAttribute("aria-pressed", String(isVoiceMode));
  voiceModeButton.setAttribute(
    "aria-label",
    isVoiceMode ? "Stop voice mode" : "Start voice mode"
  );
  voiceModeButton.title = isVoiceMode ? "Stop voice mode" : "Start voice mode";
  setBusy(state.isBusy);
}

function startSpeechCapture({ maxDurationMs = 15000 } = {}) {
  if (!canRecognizeSpeech) {
    throw new Error("Voice input is not supported in this browser. Use Chrome or Edge.");
  }

  const recognition = new BrowserSpeechRecognition();
  let transcript = "";
  let timeoutId = null;
  let isCanceled = false;
  let isStopped = false;

  recognition.lang = VOICE_LANGUAGE_CODE;
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  const cleanup = () => {
    window.clearTimeout(timeoutId);
  };

  const stop = () => {
    if (isStopped) return;
    isStopped = true;
    recognition.stop();
  };

  const cancel = () => {
    isCanceled = true;
    stop();
  };

  const promise = new Promise((resolve, reject) => {
    recognition.addEventListener("result", (event) => {
      transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript || "")
        .join(" ")
        .trim();
    });

    recognition.addEventListener("error", (event) => {
      cleanup();
      if (isCanceled || event.error === "aborted") {
        resolve("");
        return;
      }
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        reject(new Error("Microphone permission is blocked. Allow microphone access and try again."));
        return;
      }
      if (event.error === "no-speech") {
        resolve("");
        return;
      }
      reject(new Error(`Voice input failed: ${event.error}.`));
    });

    recognition.addEventListener("end", () => {
      cleanup();
      resolve(isCanceled ? "" : transcript);
    });
  });

  recognition.start();
  timeoutId = window.setTimeout(stop, maxDurationMs);

  return { cancel, promise, stop };
}

function speakText(text) {
  if (!canSpeak) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const utterance = new SpeechSynthesisUtterance(text);
    const finish = () => {
      state.currentAudio = null;
      state.currentPlaybackResolve = null;
      resolve();
    };

    utterance.lang = VOICE_LANGUAGE_CODE;
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.addEventListener("end", finish);
    utterance.addEventListener("error", () => {
      state.currentAudio = null;
      state.currentPlaybackResolve = null;
      reject(new Error("Could not play the voice reply."));
    });

    window.speechSynthesis.cancel();
    state.currentAudio = utterance;
    state.currentPlaybackResolve = finish;
    window.speechSynthesis.speak(utterance);
  });
}

async function askAssistant(text, { focusInput = true } = {}) {
  addMessage("user", text);
  messageEl.value = "";
  autoResizeInput();
  setBusy(true);
  setVoiceStatus("Thinking...");

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
    setVoiceStatus("Ready");
    return body.answer;
  } catch (error) {
    addMessage("error", error.message);
    setVoiceStatus("Request failed.");
    throw error;
  } finally {
    setBusy(false);
    if (focusInput) {
      messageEl.focus();
    }
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const text = messageEl.value.trim();
  if (!text) return;
  await askAssistant(text);
}

async function toggleDictation() {
  if (state.isDictating && state.activeCapture) {
    state.activeCapture.stop();
    return;
  }

  try {
    setDictating(true);
    setVoiceStatus("Listening...");
    state.activeCapture = startSpeechCapture({ maxDurationMs: 15000 });
    const transcript = await state.activeCapture.promise;
    state.activeCapture = null;
    setDictating(false);

    if (!transcript) {
      setVoiceStatus("No speech detected. Try again.");
      return;
    }
    const existingText = messageEl.value.trim();
    messageEl.value = existingText ? `${existingText} ${transcript}` : transcript;
    autoResizeInput();
    setVoiceStatus("Ready");
    messageEl.focus();
  } catch (error) {
    state.activeCapture = null;
    setDictating(false);
    setVoiceStatus(error.message);
  }
}

function stopVoiceMode() {
  setVoiceMode(false);
  if (state.activeCapture) {
    state.activeCapture.cancel();
    state.activeCapture = null;
  }
  if (state.currentAudio) {
    window.speechSynthesis.cancel();
    state.currentAudio = null;
  }
  if (state.currentPlaybackResolve) {
    state.currentPlaybackResolve();
  }
  setVoiceStatus("Ready");
}

async function runVoiceMode() {
  while (state.isVoiceMode) {
    try {
      setVoiceStatus("Listening...");
      state.activeCapture = startSpeechCapture({ maxDurationMs: 20000 });
      const transcript = await state.activeCapture.promise;
      state.activeCapture = null;

      if (!state.isVoiceMode) break;
      if (!transcript) continue;

      const answer = await askAssistant(transcript, { focusInput: false });
      if (!state.isVoiceMode) break;

      setVoiceStatus("Speaking...");
      if (!state.isVoiceMode) break;
      await speakText(answer);
    } catch (error) {
      addMessage("error", error.message);
      stopVoiceMode();
      break;
    }
  }
}

function toggleVoiceMode() {
  if (state.isVoiceMode) {
    stopVoiceMode();
    return;
  }

  setVoiceMode(true);
  window.setTimeout(runVoiceMode, 0);
}

async function diagnoseImage() {
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
messageEl.addEventListener("input", autoResizeInput);
diagnoseButton.addEventListener("click", diagnoseImage);
attachButton.addEventListener("click", () => imageInput.click());
micButton.addEventListener("click", toggleDictation);
voiceModeButton.addEventListener("click", toggleVoiceMode);
imageInput.addEventListener("change", () => {
  if (imageInput.files[0]) {
    diagnosisEl.textContent = `${imageInput.files[0].name} selected.`;
  }
});

if (!canRecognizeSpeech) {
  micButton.disabled = true;
  voiceModeButton.disabled = true;
  micButton.title = "Voice input works in Chrome or Edge on localhost/HTTPS.";
  voiceModeButton.title = "Voice mode works in Chrome or Edge on localhost/HTTPS.";
  setVoiceStatus("Voice input is not supported here. Use Chrome or Edge on localhost/HTTPS.");
} else {
  setVoiceStatus("Ready");
}

addMessage(
  "assistant",
  "Tell me what you see in the crop, or upload a wheat leaf image for disease detection."
);
