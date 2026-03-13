function goHome() {
  window.location.href = "index.html";
}
function goProg() {
  window.location.href = "progress.html";
}
function logout() {
  localStorage.removeItem('user');
  sessionStorage.removeItem('user');
  window.location.href = "home.html";
}

const quizTypeSelection = document.getElementById("quiz-type-selection");
const normalQuizSetup = document.getElementById("normal-quiz-setup");
const pdfQuizSetup = document.getElementById("pdf-quiz-setup");
const normalQuizBtn = document.getElementById("normalQuizBtn");
const pdfQuizBtn = document.getElementById("pdfQuizBtn");
const backToTypeSelect1 = document.getElementById("backToTypeSelect1");
const backToTypeSelect2 = document.getElementById("backToTypeSelect2");
const startNormalQuizBtn = document.getElementById("startNormalQuizBtn");
const startPdfQuizBtn = document.getElementById("startPdfQuizBtn");

const categorySelect = document.getElementById("categorySelect");
const difficultySelect = document.getElementById("difficultySelect");
const amountSelect = document.getElementById("amountSelect");
const timerSelect = document.getElementById("timerSelect");
const adaptiveToggle = document.getElementById("adaptiveToggle");

const customTopicInput = document.getElementById("customTopicInput");

const pdfInput = document.getElementById("pdfInput");
const pdfDifficultySelect = document.getElementById("pdfDifficultySelect");
const pdfAmountSelect = document.getElementById("pdfAmountSelect");
const pdfPreview = document.getElementById("pdfPreview");
const pdfTopicRename = document.querySelector(".pdf-topic-rename");
const pdfTopicName = document.getElementById("pdfTopicName");

const quizSection = document.querySelector(".quiz-section");
const questionEl = document.getElementById("question");
const optionsEl = document.querySelector(".options");
const nextBtn = document.getElementById("next");
const explainBtn = document.getElementById("explain");
const explanationEl = document.getElementById("explanation");
const resultEl = document.getElementById("result");
const timerEl = document.getElementById("timer");
const progressEl = document.querySelector(".progress");
const questionCounter = document.getElementById("questionCounter");
const correctSound = document.getElementById("correctSound");
const wrongSound = document.getElementById("wrongSound");
const restartBtn = document.getElementById("restartQuiz");
const goToProgressBtn = document.getElementById("goToProgressBtn");

const performanceDashboard = document.getElementById("performanceDashboard");
const accuracyStat = document.getElementById("accuracyStat");
const avgTimeStat = document.getElementById("avgTimeStat");
const badgesEarned = document.getElementById("badgesEarned");

let quizToken = null;
let questions = [];
let currentIndex = 0;
let score = 0;
let timer = null;
let currentCorrectAnswer = "";
let quizType = null;
let totalTimeTaken = 0;
let timerStart = null;


// Load categories with "Others" option
async function loadCategories() {
  const res = await fetch('https://opentdb.com/api_category.php');
  const data = await res.json();
  data.trivia_categories.forEach(cat => {
    const opt = document.createElement("option");
    opt.value = cat.id;
    opt.textContent = cat.name;
    categorySelect.appendChild(opt);
  });
  const othersOpt = document.createElement("option");
  othersOpt.value = "-1";
  othersOpt.textContent = "Others (Custom Topic)";
  categorySelect.appendChild(othersOpt);
}

categorySelect.addEventListener("change", () => {
  if (categorySelect.value === "-1") {
    customTopicInput.classList.remove("hidden");
  } else {
    customTopicInput.classList.add("hidden");
    customTopicInput.value = "";
  }
});


async function getToken() {
  const res = await fetch('https://opentdb.com/api_token.php?command=request');
  const data = await res.json();
  quizToken = data.token;
}

normalQuizBtn.onclick = () => {
  quizType = "normal";
  quizTypeSelection.classList.add("hidden");
  normalQuizSetup.classList.remove("hidden");
};

pdfQuizBtn.onclick = () => {
  quizType = "pdf";
  quizTypeSelection.classList.add("hidden");
  pdfQuizSetup.classList.remove("hidden");
  pdfTopicRename.classList.add("hidden");
  pdfTopicName.value = "";
};

backToTypeSelect1.onclick = () => {
  normalQuizSetup.classList.add("hidden");
  quizTypeSelection.classList.remove("hidden");
  customTopicInput.classList.add("hidden");
};

backToTypeSelect2.onclick = () => {
  pdfQuizSetup.classList.add("hidden");
  quizTypeSelection.classList.remove("hidden");
  pdfPreview.src = "";
  pdfPreview.classList.add("hidden");
  startPdfQuizBtn.disabled = true;
};

pdfInput.addEventListener("change", () => {
  if (pdfInput.files.length === 0) {
    startPdfQuizBtn.disabled = true;
    pdfPreview.src = "";
    pdfPreview.classList.add("hidden");
    pdfTopicRename.classList.add("hidden");
    return;
  }
  startPdfQuizBtn.disabled = false;
  const fileURL = URL.createObjectURL(pdfInput.files[0]);
  pdfPreview.src = fileURL;
  pdfPreview.classList.remove("hidden");

  // Show the rename input so the user can edit the PDF-derived topic name
  pdfTopicRename.classList.remove("hidden");
  const fileName = pdfInput.files[0].name.replace(/\.[^/.]+$/, "");
  pdfTopicName.value = fileName;
});

startNormalQuizBtn.onclick = async () => {
  if (!quizToken) await getToken();

  if (categorySelect.value === "-1") {
    const topic = customTopicInput.value.trim();
    if (!topic) {
      alert("Please enter a custom topic.");
      return;
    }
  questions = await generateQuizForTopic(topic, parseInt(amountSelect.value), difficultySelect.value, /*isTopic=*/ true);
    document.getElementById("quiz-title").textContent = `Custom Quiz: ${topic}`;
    startQuiz();
    return;
  }

  const category = categorySelect.value;
  const difficulty = difficultySelect.value;
  const amount = amountSelect.value;
  const categoryName = categorySelect.options[categorySelect.selectedIndex].text;
  document.getElementById("quiz-title").textContent = `${categoryName} Quiz`;

  let apiUrl = `https://opentdb.com/api.php?amount=${amount}&token=${quizToken}&type=multiple`;
  if (category) apiUrl += `&category=${category}`;
  if (difficulty) apiUrl += `&difficulty=${difficulty}`;

  const res = await fetch(apiUrl);
  const data = await res.json();

  if (data.response_code === 4) {
    await fetch(`https://opentdb.com/api_token.php?command=reset&token=${quizToken}`);
    return startNormalQuizBtn.click();
  }
  questions = data.results;
  startQuiz();
}

async function generateQuizForTopic(topic, amount, difficulty, isTopic=false) {
  // Prefer server-side generation when available. Server will choose online/offline mode.
  if (navigator.onLine) {
    try {
      const res = await fetch('/generate-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: topic, amount: amount, difficulty: difficulty, is_topic: !!isTopic })
      });
      if (res.ok) {
        const data = await res.json();
        const quiz = data.quiz || [];
        // Normalize server format (question, options, answer) -> frontend format
        return quiz.map((it) => {
          if (it.question && it.options && it.answer) {
            return {
              question: it.question,
              correct_answer: it.answer,
              incorrect_answers: it.options.filter(o => o !== it.answer)
            };
          }
          return it; // unknown shape, return as-is
        });
      }
    } catch (err) {
      console.warn('Server quiz generation failed, falling back to client generator', err);
    }
  }
  // Fallback: simple client-side generator (original behavior)
  const base = `Questions generated for topic "${topic}" with difficulty ${difficulty || "any"}.`;
  const generated = [];
  for (let i = 1; i <= amount; i++) {
    generated.push({
      question: `${base} Question ${i}?`,
      correct_answer: `Answer ${i}`,
      incorrect_answers: [`Wrong ${i}A`, `Wrong ${i}B`, `Wrong ${i}C`],
    });
  }
  return generated;
}

startPdfQuizBtn.onclick = async () => {
  if (pdfInput.files.length === 0) {
    alert("Please upload a PDF file first.");
    return;
  }
  const difficulty = pdfDifficultySelect.value;
  const amount = parseInt(pdfAmountSelect.value);
  const topicName = pdfTopicName.value.trim() || "PDF Quiz";

  document.getElementById("quiz-title").textContent = topicName;

  const file = pdfInput.files[0];
  const fileReader = new FileReader();

  fileReader.onload = function () {
    const typedarray = new Uint8Array(this.result);
    const loadingTask = pdfjsLib.getDocument(typedarray);
    loadingTask.promise.then(async function (pdf) {
      let fullText = "";
      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        const page = await pdf.getPage(pageNum);
        const content = await page.getTextContent();
        const strings = content.items.map(item => item.str);
        fullText += strings.join(" ") + " ";
      }
      questions = await questionGenerator(fullText, difficulty, amount);
      if (!questions || questions.length === 0) {
        alert("No questions could be generated from the PDF.");
        return;
      }
      startQuiz();
    }, function (reason) {
      alert("Error loading PDF: " + reason);
    });
  };

  fileReader.readAsArrayBuffer(file);
}

async function questionGenerator(text, difficulty, amount) {
  // Prefer server-side generation when online; server will use GEMINI API if configured
  if (navigator.onLine) {
    try {
      const res = await fetch('/generate-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, amount: amount, difficulty: difficulty })
      });
        if (res.ok) {
          const data = await res.json();
          const quiz = data.quiz || [];
          return quiz.map((it) => {
            if (it.question && it.options && it.answer) {
              return {
                question: it.question,
                correct_answer: it.answer,
                incorrect_answers: it.options.filter(o => o !== it.answer)
              };
            }
            return it;
          });
        }
    } catch (err) {
      console.warn('Server PDF quiz generation failed, falling back to client parser', err);
    }
  }

  // Fallback local extraction when server not reachable
  const sentences = text.split(/[.?!]\s/).filter(s => s.length > 20);
  const generated = sentences.slice(0, amount).map((sentence, index) => ({
    question: `What is the topic of sentence ${index + 1}?`,
    correct_answer: sentence,
    incorrect_answers: ["Option 1", "Option 2", "Option 3"],
  }));
  return generated;
}

function startQuiz() {
  currentIndex = 0;
  score = 0;
  totalTimeTaken = 0;
  document.querySelectorAll(".quiz-setup").forEach(el => el.classList.add("hidden"));
  quizSection.classList.remove("hidden");
  performanceDashboard.classList.add("hidden");
  goToProgressBtn.classList.add("hidden");
  showQuestion();
}

function decodeHTML(html) {
  const txt = document.createElement("textarea");
  txt.innerHTML = html;
  return txt.value;
}

function showQuestion() {
  if (currentIndex >= questions.length) {
    showResult();
    return;
  }
  const q = questions[currentIndex];
  questionEl.textContent = decodeHTML(q.question || "");
  optionsEl.innerHTML = "";
  explanationEl.classList.add("hidden");
  explainBtn.classList.add("hidden");
  nextBtn.disabled = true;
  resultEl.classList.add("hidden");

  const opts = (q.incorrect_answers && q.correct_answer)
    ? [...q.incorrect_answers, q.correct_answer]
    : (q.options || []);

  const options = opts.map(decodeHTML).sort(() => Math.random() - 0.5);
  currentCorrectAnswer = decodeHTML(q.correct_answer || q.answer || "");

  options.forEach(option => {
    const btn = document.createElement("button");
    btn.textContent = option;
    btn.onclick = () => selectAnswer(btn);
    optionsEl.appendChild(btn);
  });

  questionCounter.textContent = `Question ${currentIndex + 1} of ${questions.length}`;
  const selectedTime = parseInt(timerSelect.value);
  startTimer(selectedTime);
  updateProgress();
  timerStart = Date.now();
}

function selectAnswer(btn) {
  const userAnswer = btn.textContent;
  const allButtons = optionsEl.querySelectorAll("button");
  allButtons.forEach(b => {
    b.disabled = true;
    if (b.textContent === currentCorrectAnswer) {
      b.classList.add("correct");
    }
    if (b === btn && userAnswer !== currentCorrectAnswer) {
      b.classList.add("wrong");
    }
  });

  const timeTaken = Math.floor((Date.now() - timerStart) / 1000);
  totalTimeTaken += timeTaken;

  if (userAnswer === currentCorrectAnswer) {
    score++;
    correctSound.play();
  } else {
    wrongSound.play();
    if (adaptiveToggle && adaptiveToggle.checked && difficultySelect.value !== "easy") {
      const currentDifficulty = difficultySelect.value;
      if (currentDifficulty === "hard") difficultySelect.value = "medium";
      else if (currentDifficulty === "medium") difficultySelect.value = "easy";
    }
  }
  explainBtn.classList.remove("hidden");
  nextBtn.disabled = false;
  clearInterval(timer);
}

function startTimer(seconds) {
  let time = seconds;
  timerEl.textContent = time;
  clearInterval(timer);

  if (seconds === 0) {
    timerEl.textContent = "∞";
    return;
  }

  timer = setInterval(() => {
    time--;
    timerEl.textContent = time;
    if (time <= 0) {
      clearInterval(timer);
      nextQuestion();
    }
  }, 1000);
}

function updateProgress() {
  const progress = ((currentIndex + 1) / questions.length) * 100;
  progressEl.style.width = `${progress}%`;

  if (progress < 30) {
    questionCounter.textContent += " • Keep going!";
  } else if (progress < 70) {
    questionCounter.textContent += " • You're doing great!";
  } else {
    questionCounter.textContent += " • Almost there!";
  }
}

nextBtn.addEventListener("click", nextQuestion);

function nextQuestion() {
  currentIndex++;
  if (currentIndex < questions.length) {
    showQuestion();
  } else {
    showResult();
  }
}

explainBtn.addEventListener("click", async () => {
  // Call backend explanation endpoint when possible
  const q = questions[currentIndex];
  const questionText = q.question || q.question_text || "";
  const correct = q.correct_answer || q.answer || q.correct || "";
  explanationEl.textContent = "Loading explanation...";
  explanationEl.classList.remove("hidden");
  if (navigator.onLine) {
    try {
      const res = await fetch('/explain-answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText, correct_answer: correct })
      });
      if (res.ok) {
        const data = await res.json();
        explanationEl.textContent = data.explanation || data.explain || "No explanation returned.";
        return;
      }
    } catch (err) {
      console.warn('Explain API failed, showing local message', err);
    }
  }
  explanationEl.textContent = "Explanation functionality not available (offline).";
});

function showResult() {
  // Hide all quiz setup and quiz sections
  document.getElementById('quiz-type-selection').classList.add('hidden');
  document.getElementById('normal-quiz-setup').classList.add('hidden');
  document.getElementById('pdf-quiz-setup').classList.add('hidden');
  document.querySelector('.quiz-section').classList.add('hidden');

  // Show only the performance dashboard
  document.getElementById('performanceDashboard').classList.remove('hidden');

  // Change main heading to "Your Performance"
  document.getElementById('quiz-title').textContent = "Your Performance";

  // Your usual stats logic below
  const accuracy = Math.round((score / questions.length) * 100);
  const avgTime = Math.round(totalTimeTaken / questions.length);

  accuracyStat.textContent = `${accuracy}%`;
  avgTimeStat.textContent = `${avgTime}s`;

  let badge = "None";
  if (accuracy === 100) badge = "Perfect!";
  else if (accuracy >= 80) badge = "Gold";
  else if (accuracy >= 50) badge = "Silver";
  else badge = "Keep Practicing";
  badgesEarned.textContent = badge;
  // Record quiz result automatically in progress tracker only for PDF quizzes
  if (quizType === 'pdf') {
    try {
      const titleText = document.getElementById('quiz-title').textContent || '';
      const iso = new Date().toISOString();
      const topic = (pdfTopicName && pdfTopicName.value.trim()) ? pdfTopicName.value.trim() : titleText;
      const quizName = (pdfTopicName && pdfTopicName.value.trim()) ? pdfTopicName.value.trim() : `${titleText} - ${iso}`;
      const quizRecord = { quizName, topic, score: accuracy, date: iso.slice(0,10) };
      // Immediately persist locally so progress page will always show the result
      try {
        const raw = localStorage.getItem('progressData');
        let payload = { subjects: [], quizzes: [] };
        if (raw) {
          try { payload = JSON.parse(raw); } catch(e) { payload = { subjects: [], quizzes: [] }; }
        }
        payload.quizzes = payload.quizzes || [];
        const found = payload.quizzes.findIndex(q => q.quizName === quizRecord.quizName);
        if (found >= 0) payload.quizzes[found] = quizRecord; else payload.quizzes.push(quizRecord);
        localStorage.setItem('progressData', JSON.stringify(payload));
      } catch (e) {
        console.warn('Could not write local progress before sync', e);
      }
      // Then attempt to sync to server (async)
      recordQuizResult(quizRecord);
    } catch (e) { console.warn('Failed to record quiz result', e); }
  }
}

// Persist quiz result to server or localStorage fallback (structure matches progress page)
async function recordQuizResult(quizRecord) {
  // Try to post to server via sync-progress endpoint which accepts full payload
  const trySync = async (payload) => {
    try {
      const res = await fetch('/api/sync-progress', {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      return res.ok;
    } catch (e) {
      return false;
    }
  };

  // Attempt to fetch existing local data and append
  try {
    // If server available and online, attempt a sync by fetching current progress then updating
    if (navigator.onLine) {
      try {
        const getRes = await fetch('/api/user-progress', {credentials: 'include'});
        if (getRes.ok) {
          const data = await getRes.json();
          data.quizzes = data.quizzes || [];
          // Upsert by quizName on server copy
          const idx = data.quizzes.findIndex(q => q.quizName === quizRecord.quizName);
          if (idx >= 0) data.quizzes[idx] = quizRecord; else data.quizzes.push(quizRecord);
          const ok = await trySync(data);
          if (ok) return;
        }
      } catch (e) {
        // fallthrough to local save
      }
    }
    // Save locally
    const raw = localStorage.getItem('progressData');
    let payload = { subjects: [], quizzes: [] };
    if (raw) {
      try { payload = JSON.parse(raw); } catch(e) { payload = { subjects: [], quizzes: [] }; }
    }
    const idx2 = payload.quizzes.findIndex(q => q.quizName === quizRecord.quizName);
    if (idx2 >= 0) payload.quizzes[idx2] = quizRecord; else payload.quizzes.push(quizRecord);
    localStorage.setItem('progressData', JSON.stringify(payload));
  } catch (e) {
    console.error('recordQuizResult error', e);
  }
}


goToProgressBtn.addEventListener("click", () => {
  goProg();
});

// Keep the in-quiz restart behaviour (reload) intact if the element exists
if (typeof restartBtn !== 'undefined' && restartBtn) {
  // existing listener already attached earlier; no override here
} else {
  const rb = document.getElementById("restartQuiz");
  if (rb) rb.addEventListener("click", () => location.reload());
}

// Safe navigation handlers for performance dashboard buttons
const restartPerfBtn = document.getElementById("restartQuizPerf");
if (restartPerfBtn) restartPerfBtn.onclick = () => { window.location.href = "index.html"; };
const goToProgressPerfBtn = document.getElementById("goToProgressPerf");
if (goToProgressPerfBtn) goToProgressPerfBtn.onclick = () => { window.location.href = "progress.html"; };
const goToProgressBtnEl = document.getElementById("goToProgressBtn");
if (goToProgressBtnEl) goToProgressBtnEl.onclick = () => { window.location.href = "progress.html"; };
// Enhance button hover and keyboard focus for better UX

document.addEventListener('DOMContentLoaded', () => {
  loadCategories();
  [normalQuizBtn,pdfQuizBtn,startNormalQuizBtn,startPdfQuizBtn].forEach(btn => {
     if(btn) btn.onkeyup = (e) => { if(e.key==="Enter" || e.key===" ") { btn.click(); }};
  });
});

// Initialize app
loadCategories();
