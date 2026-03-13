// --- Navigation ---
function goLogin() { window.location.href = "login.html"; }
function goChat() { window.location.href = "chat.html"; }
function goQuiz() { window.location.href = "quiz.html"; }
function goProgress() { window.location.href = "progress.html"; }
function goStudyRoom() { window.location.href = "study1.html"; }
function goSummarizer() {
  window.location.href = "summarizer.html";  // Adjust URL to your summarizer page
}


function goProfile() {
  window.location.href = "profile.html";
}

// FAQ accordion toggle
document.querySelectorAll('.faq-question').forEach(question => {
  question.addEventListener('click', () => {
    question.classList.toggle('active');
    const answer = question.nextElementSibling;
    if (question.classList.contains('active')) {
      answer.style.maxHeight = answer.scrollHeight + "px";
    } else {
      answer.style.maxHeight = null;
    }
  });
});
const quotes = [
  "Every step forward is progress — keep moving! 🚀",
  "Consistency beats intensity — focus on daily gains.",
  "Learn from mistakes, grow stronger every day.",
  "Small efforts build big achievements over time.",
  "Knowledge is power — stay curious and explore.",
  "You have what it takes — believe in yourself!",
  "Studying smart leads to success, not just hard work."
];

function setupMarquee() {
  const marqueeEl = document.getElementById('marqueeText');
  if (!marqueeEl) return;
  // Pick a random quote initially
  let idx = Math.floor(Math.random() * quotes.length);
  marqueeEl.textContent = quotes[idx];

  // Update quote every 20 seconds with animation sync
  setInterval(() => {
    idx = (idx + 1) % quotes.length;
    marqueeEl.textContent = quotes[idx];
  }, 20000);
}

// Call on DOM ready
document.addEventListener('DOMContentLoaded', setupMarquee);
