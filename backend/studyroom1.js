// study2.js (Complete Code with Shared AI Integration - FULL FILE)
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.4.0/firebase-app.js";
import { getDatabase, ref, push, onValue, onChildAdded, set } from "https://www.gstatic.com/firebasejs/10.4.0/firebase-database.js";

// ✅ Firebase configuration 
const firebaseConfig = {
    apiKey: "AIzaSyCsdv2DcDa_u-kNBAXuq3TXp9nfdLy-chs",
    authDomain: "margadarshi-85118.firebaseapp.com",
    databaseURL: "https://margadarshi-85118-default-rtdb.firebaseio.com",
    projectId: "margadarshi-85118",
    storageBucket: "margadarshi-85118.appspot.com",
    messagingSenderId: "868205365403",
    appId: "1:868205365403:web:0c9d02714c17537068de3e"
};

// --- Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

// --- Persistent User Storage
let userId = localStorage.getItem("userId");
let username = localStorage.getItem("username");

if (!userId) {
    userId = "user-" + Date.now();
    localStorage.setItem("userId", userId);
}
if (!username) {
    username = prompt("Enter your display name") || "User";
    localStorage.setItem("username", username);
}

// --- Global Variables
let currentRoomId = null;
let currentPasscode = null;
let messagesListenerAttached = false;
const AI_USER_ID = "AI_Study_Partner_ID"; // Unique ID for the AI

// --- Utility: Timestamp Format
function formatTimestamp(ts) {
    if (!ts) return "";
    return new Date(Number(ts)).toLocaleString("en-IN", {
        timeZone: "Asia/Kolkata",
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
    });
}

// --- DOM Elements ---
const roomModal = document.getElementById("roomModal");
const mainLayout = document.getElementById("mainLayout");
const createRoomBtn = document.getElementById("createRoomBtn");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

const groupChatBox = document.getElementById("groupChatBox");
const aiChatBox = document.getElementById("aiChatBox");
const fileInput = document.getElementById("fileInput");
const roomIdDisplay = document.getElementById("roomIdDisplay");
const passcodeDisplay = document.getElementById("passcodeDisplay");

// --- AI Tab Elements ---
const aiTab = document.getElementById("aiTab");
const groupTab = document.getElementById("groupTab");

// --- Header Controls ---
const logoutBtn = document.getElementById("logoutBtn");
const homeBtn = document.getElementById("homeBtn");

// --- Quiz & Leaderboard Elements ---
const startQuizBtn = document.getElementById("startQuizBtn");
const viewLeaderboardBtn = document.getElementById("viewLeaderboardBtn");
const quizContainer = document.getElementById("quizContainer");
const leaderboardContainer = document.getElementById("leaderboardContainer");


// --- Tab Switching Logic (Now both tabs use groupChatBox to display output) ---
if (aiTab && groupTab) {
    // When switching to AI, just hide the AI Chat Box (it's no longer used for output)
    aiTab.addEventListener("click", () => {
        aiTab.classList.add("selected");
        groupTab.classList.remove("selected");
        aiChatBox.classList.add("hidden"); // Keep hidden as we use groupChatBox for output
        groupChatBox.classList.remove("hidden"); // Show Group chat for all messages
        // Visually, the user is still on the 'Group Discussion' box, but the selected tab is 'AI'.
    });

    groupTab.addEventListener("click", () => {
        groupTab.classList.add("selected");
        aiTab.classList.remove("selected");
        groupChatBox.classList.remove("hidden");
        aiChatBox.classList.add("hidden");
    });
}

// Handle Enter key for sending messages
chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendBtn.click();
    }
});


// --- ROOM CREATION ---
if (createRoomBtn) {
    createRoomBtn.addEventListener("click", () => {
        const displayNameInput = document.getElementById("displayNameInput");
        if (!displayNameInput.value.trim()) {
            showToast("Enter display name", "#e11d48");
            return;
        }
        username = displayNameInput.value.trim();
        localStorage.setItem("username", username);

        currentRoomId = "room" + Math.floor(Math.random() * 9000 + 1000);
        currentPasscode = Math.floor(Math.random() * 9000 + 1000);

        set(ref(db, `rooms/${currentRoomId}`), { passcode: currentPasscode })
            .then(() => {
                showToast("✅ Room created successfully!");
                openRoom();
            })
            .catch((err) => console.error("Create room error:", err));
    });

}

// --- LOGOUT FUNCTIONALITY ---
if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
        localStorage.removeItem("userId");
        localStorage.removeItem("username");
        showToast("You have logged out.");
        setTimeout(() => location.reload(), 1200);
    });
}

// --- HOME BUTTON ---
if (homeBtn) {
    homeBtn.addEventListener("click", () => {
        window.location.href = "index.html";
    });
}

// --- JOIN ROOM FLOW ---
const joinRoomBtn = document.getElementById("joinRoomBtn");
const roomJoinForm = document.getElementById("roomJoinForm");
const doJoinBtn = document.getElementById("doJoinBtn");
const joinRoomIdInput = document.getElementById("joinRoomId");
const joinPasscodeInput = document.getElementById("joinPasscode");

if (joinRoomBtn && roomJoinForm) {
    joinRoomBtn.addEventListener("click", () => {
        roomJoinForm.classList.toggle("hidden");
    });
}

if (doJoinBtn) {
    doJoinBtn.addEventListener("click", () => {
        const displayNameInput = document.getElementById("displayNameInput");
        if (!displayNameInput.value.trim()) {
            showToast("Enter display name", "#e11d48");
            return;
        }
        username = displayNameInput.value.trim();
        localStorage.setItem("username", username);

        const roomId = joinRoomIdInput.value.trim();
        const passcode = joinPasscodeInput.value.trim();
        if (!roomId || !passcode) return showToast("Enter Room ID and Passcode", "#e11d48");
        joinRoom(roomId, passcode);
    });

}

// --- JOIN ROOM FUNCTION ---
function joinRoom(roomId, passcode) {
    const roomRef = ref(db, `rooms/${roomId}/passcode`);
    onValue(
        roomRef,
        (snapshot) => {
            if (!snapshot.exists()) return showToast("❌ Room not found", "#e11d48");
            if (snapshot.val() != passcode) return showToast("Wrong passcode", "#e11d48");

            currentRoomId = roomId;
            currentPasscode = passcode;
            showToast("✅ Joined the room successfully!");
            openRoom();
        }, { onlyOnce: true }
    );
}

// --- OPEN ROOM UI ---
function openRoom() {
    roomModal.classList.add("hidden");
    mainLayout.classList.remove("hidden");

    roomIdDisplay.innerText = "Room ID: " + currentRoomId;
    passcodeDisplay.innerText = "Passcode: " + currentPasscode;

    if (window.history.replaceState) {
        const cleanUrl = window.location.origin + window.location.pathname;
        window.history.replaceState(null, "", cleanUrl);
    }

    const roomLinkDiv = document.getElementById("roomLinkDiv");
    if (roomLinkDiv) {
        const currentJoinLink = `${window.location.origin}${window.location.pathname}?roomId=${currentRoomId}&passcode=${currentPasscode}`;
        roomLinkDiv.innerHTML = `
      <b>Room Link:</b> 
      <a href="${currentJoinLink}" target="_blank">Join Room</a>
      <button id="copyJoinLinkBtn">Copy Link</button>`;
        const copyJoinLinkBtn = document.getElementById("copyJoinLinkBtn");
        copyJoinLinkBtn.addEventListener("click", () => {
            navigator.clipboard.writeText(currentJoinLink).then(() => showToast("🔗 Room link copied!"));
        });
    }

    groupChatBox.innerHTML = "";

    if (!messagesListenerAttached) {
        listenMessages();
        messagesListenerAttached = true;
    }
}

// ⬇️ MODIFIED CODE BLOCK START ⬇️
// --- SEND MESSAGE (MODIFIED TO HANDLE SHARED AI & FILES) ---
if (sendBtn) {
    sendBtn.addEventListener("click", async() => {
        const text = chatInput.value.trim();
        const file = fileInput.files[0];

        if (!text && !file) return;

        if (aiTab.classList.contains("selected")) {
            // 1. Process as SHARED AI Request
            if (!currentRoomId) {
                return showToast("Please join a room to use the collaborative AI.", "#e11d48");
            }

            let filePayload = null;
            if (file) {
                if (file.size > 2 * 1024 * 1024) { // Guardrail: Limit file size to 2MB for stability
                    return showToast("File is too large for AI processing (max 2MB).", "#e11d48");
                }

                // Read file into Base64 
                const reader = new FileReader();
                reader.readAsDataURL(file);
                await new Promise(resolve => { reader.onload = resolve; });

                filePayload = {
                    data: reader.result, // Base64 dataURL
                    mimeType: file.type
                };
            }

            if (!text && !filePayload) {
                return showToast("Enter a prompt or select a file for the AI.", "#e11d48");
            }

            // 1a. User's Request: Push to Firebase for all to see
            const userMessage = {
                sender: userId,
                username: username,
                // ⭐ THE CRITICAL FIX IS HERE: Ensures text takes precedence
                text: text ? text : (file ? `[Shared File: ${file.name}]` : `[No text sent]`),
                timestamp: Date.now(),
            };
            // Add file details to the Firebase message if present (for display)
            if (filePayload) {
                userMessage.fileName = file.name;
                userMessage.fileType = file.type;
                userMessage.fileData = filePayload.data;
            }
            push(ref(db, `rooms/${currentRoomId}/messages`), userMessage);

            chatInput.value = ""; // Clear input immediately
            fileInput.value = ""; // Clear file input immediately

            // 1b. Display a temporary "AI Thinking" message
            const thinkingMsg = {
                sender: AI_USER_ID,
                username: "AI Study Partner",
                text: "Thinking...",
                timestamp: Date.now()
            };
            displayMessage(groupChatBox, thinkingMsg);
            const thinkingDiv = groupChatBox.lastChild;

            // 1c. Get the real AI response from the server (pass file info)
            const aiResponseText = await sendQueryToGemini(text, filePayload);

            // 1d. Remove the local 'Thinking...' message 
            if (thinkingDiv) thinkingDiv.remove();

            // 1e. AI Response: Push the final AI Response to Firebase for all to see
            const aiMessage = {
                sender: AI_USER_ID,
                username: "🤖 AI Partner",
                text: aiResponseText,
                timestamp: Date.now(),
            };
            push(ref(db, `rooms/${currentRoomId}/messages`), aiMessage);


        } else if (currentRoomId) {
            // 2. Process as Group Chat Message (Existing Logic)
            if (fileInput.files.length > 0) {
                // File upload logic handled by fileInput listener (no change)
            } else if (text) {
                push(ref(db, `rooms/${currentRoomId}/messages`), {
                    sender: userId,
                    username: username,
                    text: text,
                    timestamp: Date.now()
                }).catch((err) => console.error("Push message error:", err));
            }
        } else {
            showToast("Join or create a room for group chat.", "#e11d48");
        }

        // Clear input after processing non-AI message
        if (!aiTab.classList.contains("selected")) {
            chatInput.value = "";
            fileInput.value = "";
        }
    });
}
// ⬆️ MODIFIED CODE BLOCK END ⬆️


// --- Unified Message Display Function ---
function displayMessage(chatBoxElement, msg) {
    const div = document.createElement("div");

    // Determine the class based on sender (AI, You, or Other)
    if (msg.sender === userId) {
        div.className = "msg me";
    } else if (msg.sender === AI_USER_ID) {
        div.className = "msg ai";
    } else {
        div.className = "msg other";
    }

    const displayName = msg.sender === userId ? "You" : (msg.sender === AI_USER_ID ? "🤖 AI Partner" : msg.username || "User");

    // Logic for rendering text or file messages
    if (msg.fileData) {
        const fileType = msg.fileType || "";
        if (fileType.startsWith("image/")) {
            div.innerHTML = `<b>${displayName}:</b><br><img src="${msg.fileData}" style="max-width:400px;border-radius:8px;margin-top:6px;">`;
        } else if (fileType === "application/pdf") {
            let base64Data = msg.fileData;
            if (base64Data.startsWith("data:")) {
                base64Data = base64Data.split(",")[1];
            }
            const blob = b64toBlob(base64Data, fileType);
            const url = URL.createObjectURL(blob);
            div.innerHTML = `<b>${displayName}:</b> <a href="${url}" target="_blank">${msg.fileName || "PDF File"}</a>`;
        } else {
            div.innerHTML = `<b>${displayName}:</b> <a href="${msg.fileData}" download="${msg.fileName || "File"}">${msg.fileName || "File"}</a>`;
        }
    } else if (msg.text) {
        div.innerHTML = `<b>${displayName}:</b> ${msg.text}`;
    }

    const timeEl = document.createElement("div");
    timeEl.className = "msg-time";
    timeEl.style.fontSize = "0.75em";
    timeEl.style.color = "#666";
    timeEl.innerText = formatTimestamp(msg.timestamp);
    div.appendChild(timeEl);

    chatBoxElement.appendChild(div);
    chatBoxElement.scrollTop = chatBoxElement.scrollHeight;
}


// 🐍 Python-Gemini AI Integration Functions (Connects to server.py) 🐍
// ---------------------------------------------------------------------

// --- Utility: Convert Data URL to Base64 String (Removes 'data:mime/type;base64,') ---
function cleanDataURL(dataUrl) {
    // This is robust against data URL formats
    const parts = dataUrl.split(',');
    return parts.length > 1 ? parts[1] : dataUrl;
}


// --- Function to communicate with your Python backend server (server.py) ---
async function sendQueryToGemini(query, filePayload = null) {
    // This URL MUST match the address and port of your running Python server (Flask)
    const serverUrl = 'http://127.0.0.1:3000/ask-ai';

    try {
        const payload = {
            prompt: query,
            roomId: currentRoomId,
            file: filePayload ? {
                data: cleanDataURL(filePayload.data), // Clean Base64 string
                mimeType: filePayload.mimeType
            } : null
        };

        const response = await fetch(serverUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            // SENDING THE ROOM ID, PROMPT, AND FILE (IF PRESENT)
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown server error.' }));
            throw new Error(`Server Error (${response.status}): ${errorData.error}`);
        }

        const data = await response.json();
        return data.text;

    } catch (error) {
        console.error('Error contacting AI server:', error);
        return "Sorry, the collaborative AI Study Partner is offline. Please ask the room host to check the server status!";
    }
}
// ---------------------------------------------------------------------

// --- LISTEN FOR MESSAGES ---
function listenMessages() {
    if (!currentRoomId) return;

    const messagesRef = ref(db, `rooms/${currentRoomId}/messages`);
    // Listen for new messages pushed to the room
    onChildAdded(messagesRef, (snapshot) => {
        const msg = snapshot.val();
        if (!msg) return;
        displayMessage(groupChatBox, msg);
    });
}

// --- FILE UPLOAD ---
if (fileInput) {
    fileInput.addEventListener("change", async() => {
        const file = fileInput.files[0];
        if (!file || !currentRoomId) return showToast("Please join a room to share files.", "#e11d48");

        // The AI tab logic now handles file upload and message creation for AI mode.
        // This block handles regular group chat file sharing.
        if (aiTab.classList.contains("selected")) {
            // Do nothing here, the sendBtn handler will manage the AI file
            return;
        }

        const reader = new FileReader();
        reader.onload = async function() {
            const base64Data = reader.result;
            await push(ref(db, `rooms/${currentRoomId}/messages`), {
                sender: userId,
                username: username,
                fileName: file.name,
                fileType: file.type,
                fileData: base64Data,
                timestamp: Date.now()
            });
            showToast("📎 File shared successfully!");
            fileInput.value = "";
        };
        reader.readAsDataURL(file);
    });
}

// --- Base64 → Blob (Utility for downloading files) ---
function b64toBlob(b64Data, contentType = "", sliceSize = 512) {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];

    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
        const slice = byteCharacters.slice(offset, offset + sliceSize);
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
    }

    return new Blob(byteArrays, { type: contentType });
}

// --- Toast Notification ---
function showToast(msg, color = "#2563eb") {
    let toast = document.createElement("div");
    toast.innerText = msg;
    toast.style.position = "fixed";
    toast.style.bottom = "20px";
    toast.style.right = "20px";
    toast.style.background = color;
    toast.style.color = "#fff";
    toast.style.padding = "12px 16px";
    toast.style.borderRadius = "8px";
    toast.style.boxShadow = "0 3px 10px rgba(0,0,0,0.25)";
    toast.style.zIndex = "1000";
    toast.style.fontWeight = "600";
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2500);
}

// --- AUTO JOIN VIA URL ---
window.addEventListener("load", () => {
    const params = new URLSearchParams(window.location.search);
    const roomId = params.get("roomId");
    const passcode = params.get("passcode");
    if (roomId && passcode) joinRoom(roomId, passcode);
});


// --- QUIZ & LEADERBOARD LOGIC ---
if (startQuizBtn) {
    startQuizBtn.addEventListener("click", () => {
        leaderboardContainer.classList.add("hidden");
        quizContainer.classList.remove("hidden");
        showToast("Quiz feature selected (implementation pending).", "#f59e0b");
    });
}

if (viewLeaderboardBtn) {
    viewLeaderboardBtn.addEventListener("click", () => {
        quizContainer.classList.add("hidden");
        leaderboardContainer.classList.remove("hidden");
        loadLeaderboard();
    });
}

function loadLeaderboard() {
    if (!currentRoomId) return;
    const leaderboardRef = ref(db, `rooms/${currentRoomId}/quizResults`);
    onValue(leaderboardRef, (snapshot) => {
        const scores = snapshot.val() || {};
        const sortedScores = Object.values(scores).sort((a, b) => b.score - a.score);

        let leaderboardHTML = "<h3>🏆 Leaderboard</h3><ol>";
        if (sortedScores.length === 0) {
            leaderboardHTML += "<li>No quiz scores recorded yet.</li>";
        } else {
            sortedScores.forEach(userScore => {
                leaderboardHTML += `<li>${userScore.username}: <b>${userScore.score}</b> points</li>`;
            });
        }
        leaderboardHTML += "</ol>";
        leaderboardContainer.innerHTML = leaderboardHTML;
    }, { onlyOnce: true });
}