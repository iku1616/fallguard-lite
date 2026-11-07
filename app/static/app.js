"use strict";

const elements = {
  video: document.querySelector("#video"),
  overlay: document.querySelector("#overlay"),
  capture: document.querySelector("#captureCanvas"),
  stage: document.querySelector("#stage"),
  empty: document.querySelector("#emptyState"),
  loading: document.querySelector("#loadingState"),
  camera: document.querySelector("#cameraButton"),
  file: document.querySelector("#fileButton"),
  fileInput: document.querySelector("#fileInput"),
  stop: document.querySelector("#stopButton"),
  badge: document.querySelector("#connectionBadge"),
  personCount: document.querySelector("#personCount"),
  fallCount: document.querySelector("#fallCount"),
  fps: document.querySelector("#fpsLabel"),
  latency: document.querySelector("#latencyLabel"),
  events: document.querySelector("#eventList"),
  clearEvents: document.querySelector("#clearEvents"),
  alertCard: document.querySelector(".alert-card"),
};

const state = {
  socket: null,
  stream: null,
  objectUrl: null,
  running: false,
  waiting: false,
  lastSentAt: 0,
  lastFrameAt: 0,
  fpsSamples: [],
  fallenIds: new Set(),
  eventCount: 0,
};

const COLORS = {
  normal: "#33d6c5",
  suspected: "#ffbd4a",
  fallen: "#ff5c6c",
  recovering: "#8ea7ff",
};

function setBadge(kind, label) {
  elements.badge.className = `status-badge ${kind}`;
  elements.badge.querySelector("strong").textContent = label;
}

function setRunningUi(running) {
  state.running = running;
  elements.stage.classList.toggle("empty", !running);
  elements.empty.hidden = running;
  elements.camera.disabled = running;
  elements.file.disabled = running;
  elements.stop.disabled = !running;
  if (!running) {
    elements.loading.hidden = true;
    setBadge("idle", "待機中");
  }
}

function socketUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws/analyze`;
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(socketUrl());
    socket.binaryType = "arraybuffer";
    socket.addEventListener("open", () => {
      state.socket = socket;
      setBadge("live", "解析中");
      resolve();
    }, { once: true });
    socket.addEventListener("error", () => reject(new Error("解析サーバーへ接続できません")), { once: true });
    socket.addEventListener("message", handleMessage);
    socket.addEventListener("close", () => {
      if (state.running) {
        setBadge("error", "接続が切れました");
        showTransientError("解析サーバーとの接続が切れました。停止して再試行してください。");
      }
      state.waiting = false;
    });
  });
}

async function startCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("このブラウザではカメラを利用できません");
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
      audio: false,
    });
    elements.video.srcObject = state.stream;
    elements.video.controls = false;
    await elements.video.play();
    await beginAnalysis();
  } catch (error) {
    stopAnalysis();
    showTransientError(error.name === "NotAllowedError" ? "カメラの使用が許可されませんでした。" : error.message);
  }
}

async function startFile(file) {
  if (!file || !file.type.startsWith("video/")) {
    showTransientError("動画ファイルを選択してください。");
    return;
  }
  try {
    state.objectUrl = URL.createObjectURL(file);
    elements.video.srcObject = null;
    elements.video.src = state.objectUrl;
    elements.video.controls = true;
    await elements.video.play();
    await beginAnalysis();
  } catch (error) {
    stopAnalysis();
    showTransientError(`動画を開けませんでした: ${error.message}`);
  }
}

async function beginAnalysis() {
  setRunningUi(true);
  elements.loading.hidden = false;
  setBadge("idle", "接続中");
  await connectSocket();
  syncCanvasSize();
  state.lastFrameAt = performance.now();
  requestAnimationFrame(analyzeLoop);
}

function stopAnalysis() {
  state.running = false;
  state.waiting = false;
  if (state.socket) {
    state.socket.onclose = null;
    state.socket.close(1000, "client stopped");
    state.socket = null;
  }
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  elements.video.pause();
  elements.video.srcObject = null;
  elements.video.removeAttribute("src");
  elements.video.load();
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
  elements.overlay.getContext("2d").clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  elements.personCount.textContent = "0";
  elements.fallCount.textContent = "0";
  elements.alertCard.classList.remove("active");
  elements.fps.textContent = "0.0 FPS";
  elements.latency.textContent = "— ms";
  state.fallenIds.clear();
  state.fpsSamples = [];
  setRunningUi(false);
}

function syncCanvasSize() {
  const width = elements.video.videoWidth || 1280;
  const height = elements.video.videoHeight || 720;
  elements.overlay.width = width;
  elements.overlay.height = height;
  elements.stage.style.aspectRatio = `${width} / ${height}`;
}

function analyzeLoop(now) {
  if (!state.running) return;
  if (elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !elements.video.paused) {
    if (!state.waiting && state.socket?.readyState === WebSocket.OPEN && now - state.lastSentAt >= 125) {
      sendFrame(now);
    }
  }
  requestAnimationFrame(analyzeLoop);
}

function sendFrame(now) {
  const sourceWidth = elements.video.videoWidth;
  const sourceHeight = elements.video.videoHeight;
  if (!sourceWidth || !sourceHeight) return;
  const scale = Math.min(1, 960 / sourceWidth);
  elements.capture.width = Math.max(1, Math.round(sourceWidth * scale));
  elements.capture.height = Math.max(1, Math.round(sourceHeight * scale));
  const context = elements.capture.getContext("2d", { alpha: false });
  context.drawImage(elements.video, 0, 0, elements.capture.width, elements.capture.height);
  state.waiting = true;
  state.lastSentAt = now;
  elements.capture.toBlob((blob) => {
    if (!blob || !state.running || state.socket?.readyState !== WebSocket.OPEN) {
      state.waiting = false;
      return;
    }
    blob.arrayBuffer().then((buffer) => state.socket.send(buffer));
  }, "image/jpeg", 0.78);
}

function handleMessage(event) {
  state.waiting = false;
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    showTransientError("解析結果の形式が正しくありません。");
    return;
  }
  if (payload.type === "status") {
    elements.loading.hidden = payload.status !== "loading";
    if (payload.status === "loading") setBadge("idle", "モデル準備中");
    return;
  }
  if (payload.type === "error") {
    elements.loading.hidden = true;
    setBadge("error", "解析エラー");
    showTransientError(payload.message || "解析に失敗しました。");
    return;
  }
  if (payload.type !== "result") return;
  elements.loading.hidden = true;
  setBadge("live", "解析中");
  renderResult(payload);
}

function renderResult(payload) {
  const people = payload.people || [];
  const now = performance.now();
  const dt = Math.max(1, now - state.lastFrameAt);
  state.lastFrameAt = now;
  state.fpsSamples.push(1000 / dt);
  if (state.fpsSamples.length > 20) state.fpsSamples.shift();
  const avgFps = state.fpsSamples.reduce((sum, value) => sum + value, 0) / state.fpsSamples.length;
  elements.fps.textContent = `${avgFps.toFixed(1)} FPS`;
  elements.latency.textContent = `${Math.round(payload.latency_ms || 0)} ms`;
  elements.personCount.textContent = String(people.length);

  const currentFallen = new Set(people.filter((p) => p.state === "fallen").map((p) => p.track_id));
  for (const person of people) {
    if (person.state === "fallen" && !state.fallenIds.has(person.track_id)) addFallEvent(person);
  }
  state.fallenIds = currentFallen;
  elements.fallCount.textContent = String(currentFallen.size);
  elements.alertCard.classList.toggle("active", currentFallen.size > 0);
  drawPeople(people, payload.frame_width, payload.frame_height);
}

function drawPeople(people, frameWidth, frameHeight) {
  if (elements.overlay.width !== frameWidth || elements.overlay.height !== frameHeight) {
    elements.overlay.width = frameWidth;
    elements.overlay.height = frameHeight;
    elements.stage.style.aspectRatio = `${frameWidth} / ${frameHeight}`;
  }
  const context = elements.overlay.getContext("2d");
  context.clearRect(0, 0, frameWidth, frameHeight);
  context.lineJoin = "round";
  context.font = `700 ${Math.max(13, frameWidth / 75)}px Inter, sans-serif`;

  for (const person of people) {
    const [x1, y1, x2, y2] = person.bbox;
    const color = COLORS[person.state] || COLORS.normal;
    const width = Math.max(2, frameWidth / 380);
    context.strokeStyle = color;
    context.lineWidth = width;
    context.shadowColor = color;
    context.shadowBlur = person.state === "fallen" ? 14 : 5;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.shadowBlur = 0;

    const label = `#${person.track_id}  ${labelForState(person.state)}  ${Math.round((person.risk_score || 0) * 100)}%`;
    const metrics = context.measureText(label);
    const labelHeight = Math.max(24, frameWidth / 45);
    const labelY = Math.max(0, y1 - labelHeight);
    context.fillStyle = color;
    context.fillRect(x1, labelY, metrics.width + 16, labelHeight);
    context.fillStyle = person.state === "suspected" ? "#171006" : "#06110f";
    context.fillText(label, x1 + 8, labelY + labelHeight * 0.7);

    context.fillStyle = color;
    for (const point of person.keypoints || []) {
      if (point[2] >= 0.35) {
        context.beginPath();
        context.arc(point[0], point[1], Math.max(2.5, frameWidth / 300), 0, Math.PI * 2);
        context.fill();
      }
    }
  }
}

function labelForState(value) {
  return ({ normal: "通常", suspected: "要確認", fallen: "転倒の可能性", recovering: "回復確認中" })[value] || value;
}

function addFallEvent(person) {
  if (state.eventCount === 0) elements.events.replaceChildren();
  state.eventCount += 1;
  const item = document.createElement("li");
  item.className = "event-item";
  const reason = (person.reasons || []).slice(0, 2).join("・") || "姿勢と動きの条件が一致";
  item.innerHTML = `
    <span class="event-mark" aria-hidden="true">!</span>
    <span class="event-copy"><strong>人物 #${person.track_id}: 転倒の可能性</strong><small></small></span>
    <time class="event-time"></time>`;
  item.querySelector("small").textContent = reason;
  item.querySelector("time").textContent = new Intl.DateTimeFormat("ja-JP", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date());
  elements.events.prepend(item);
}

function clearEvents() {
  state.eventCount = 0;
  elements.events.innerHTML = `<li class="event-placeholder"><span aria-hidden="true">◇</span><p>転倒の疑いを検出すると<br>ここに表示されます</p></li>`;
}

function showTransientError(message) {
  setBadge("error", message);
  window.setTimeout(() => {
    if (!state.running) setBadge("idle", "待機中");
  }, 5000);
}

elements.camera.addEventListener("click", startCamera);
elements.file.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => {
  const [file] = elements.fileInput.files;
  startFile(file);
  elements.fileInput.value = "";
});
elements.stop.addEventListener("click", stopAnalysis);
elements.clearEvents.addEventListener("click", clearEvents);
elements.video.addEventListener("loadedmetadata", syncCanvasSize);
elements.video.addEventListener("ended", stopAnalysis);
window.addEventListener("beforeunload", stopAnalysis);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.stream) stopAnalysis();
});
