//progress.js
document.addEventListener('DOMContentLoaded', () => {
  // Handle Home button click - Navigate to your homepage
  const homeBtn = document.querySelector('.nav-buttons .home-btn');
  if (homeBtn) {
    homeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = 'index.html'; // Adjust URL as needed
    });
  }

  // Handle Logout button click - Redirect to logout endpoint or perform logout logic
  const logoutBtn = document.querySelector('.nav-buttons .logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', (e) => {
      e.preventDefault();
      // Simple redirect to server logout URL, modify if you have other logout logic
      window.location.href = '/logout';
    });
  }
});

document.addEventListener('DOMContentLoaded', () => {
    const userProfile = {
      targetDate: "2025-12-01",
    };

    // Start with empty subjects and quizzes, load from backend instead
    let subjects = [];
    let quizzes = [];

    const activityDates = [];

    const subjectsListEl = document.getElementById('subjects-list');
    const subjectTemplate = document.getElementById('subject-template');
    const overallFill = document.getElementById('overall-fill');
    const overallPercentEl = document.getElementById('overall-percent');
    const weakTopicsEl = document.getElementById('weak-topics');
    const quizTableBody = document.querySelector('#quiz-table tbody');
    const streakEl = document.getElementById('streak');
    const streakTextEl = document.getElementById('streak-text');
    const countdownEl = document.getElementById('countdown');
    const goalTextEl = document.getElementById('goal-text');
    const motivationEl = document.getElementById('motivation');
    const syncStatusEl = document.getElementById('syncStatus');
    const btnSave = document.getElementById('btn-save-progress');
    const btnAddSubject = document.getElementById('btn-add-subject');
    const btnGetTips = document.getElementById('btn-get-tips');
    const btnAddQuiz = document.getElementById('btn-add-quiz');

    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
    function escapeHtml(s){
      return String(s).replace(/[&<>\"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
    }
    function prettyDate(d) {
      const dt = new Date(d + 'T00:00:00');
      return dt.toLocaleDateString();
    }
     const motivationQuotes = [
    "Every little effort counts. Progress is progress!",
    "Consistency is the key to mastery.",
    "Mistakes are proof you're trying.",
    "Small steps each day lead to big results.",
    "Celebrate progress, not perfection.",
    "Your future self will thank you.",
    "Learn smarter, not harder."
  ];
  const motivationQuoteEl = document.getElementById('motivation-quote');
  const newQuoteBtn = document.getElementById('newQuoteBtn');
  newQuoteBtn.addEventListener('click', () => {
    let newQuote;
    do {
      newQuote = motivationQuotes[Math.floor(Math.random() * motivationQuotes.length)];
    } while (newQuote === motivationQuoteEl.textContent);
    motivationQuoteEl.textContent = newQuote;
  });

  // 2. Subject styling helper function
  function styleSubjectItem(subjectItem) {
    subjectItem.style.opacity = '0';
    subjectItem.style.transition = 'opacity 0.6s';
    setTimeout(() => subjectItem.style.opacity = '1', 10);
    subjectItem.addEventListener('mouseenter', () => {
      subjectItem.style.boxShadow = '0 0 15px rgba(104,109,250,0.5)';
    });
    subjectItem.addEventListener('mouseleave', () => {
      subjectItem.style.boxShadow = '';
    });
  }
    // Render subjects list with controls
    function renderSubjects() {
  subjectsListEl.innerHTML = '';
  subjects.forEach((s, idx) => {
    const node = subjectTemplate.content.cloneNode(true);
    const root = node.querySelector('.subject-item');
    root.dataset.index = idx;

    root.querySelector('.subject-name').textContent = s.name;

    const fill = root.querySelector('.progress-fill');
    const percent = s.totalChapters > 0
      ? Math.floor((s.completedChapters / s.totalChapters) * 100)
      : 0;
    fill.style.width = `${percent}%`;
    root.querySelector('.subject-progress').textContent = `${percent}%`;

    const expandBtn = root.querySelector('.expand-btn');
    const modulePanel = root.querySelector('.module-list-panel');
    const moduleList = root.querySelector('.module-list');
    let expanded = false;

    expandBtn.addEventListener('click', () => {
      expanded = !expanded;
      if (!expanded) {
        modulePanel.classList.add('hidden');
        expandBtn.innerHTML = "&#9660;";
      } else {
        renderModules(s, moduleList);
        modulePanel.classList.remove('hidden');
        expandBtn.innerHTML = "&#9650;";
      }
    });

    // Auto-expand first subject for demo
    // Auto-expand the newly added subject (last in list)
if (idx === subjects.length - 1) {
  expanded = true;
  renderModules(s, moduleList);
  modulePanel.classList.remove('hidden');
  expandBtn.innerHTML = "&#9650;";
}


    // Remove subject handler
    const removeBtn = root.querySelector('.remove-subject');
    removeBtn.addEventListener('click', () => {
      if (!confirm(`Remove subject "${s.name}"?`)) return;
      subjects.splice(idx, 1);
      renderSubjects();
      updateOverall();
      syncProgress();
    });

    subjectsListEl.appendChild(root);
  });
}

function renderModules(subject, moduleList) {
  moduleList.innerHTML = '';
  
  if (!subject.modules || !Array.isArray(subject.modules) || subject.modules.length !== subject.totalChapters) {
    subject.modules = [];
    for (let i = 0; i < subject.totalChapters; i++) {
      subject.modules.push({ name: `Module ${i+1}`, completed: false });
    }
    // Initialize completedChapters counter
    subject.completedChapters = subject.modules.filter(m => m.completed).length;
  }

  subject.modules.forEach((mod, i) => {
    const row = document.createElement('div');
    row.className = 'module-row';

    // Checkbox for completion
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'module-checkbox';
    checkbox.checked = !!mod.completed;

    checkbox.addEventListener('change', () => {
      mod.completed = checkbox.checked;
      // Recalculate completedChapters count
      subject.completedChapters = subject.modules.filter(m => m.completed).length;
      updateOverall();
      renderSubjects();
      syncProgress();
    });

    // Editable name
    const name = document.createElement('span');
    name.textContent = mod.name;
    name.className = 'module-name';
    name.title = 'Double-click to rename';

    name.addEventListener('dblclick', () => {
      const input = document.createElement('input');
      input.type = 'text';
      input.value = mod.name;
      input.className = 'module-edit';
      input.style.width = '130px';

      input.addEventListener('blur', () => {
        mod.name = input.value.trim() || mod.name;
        name.textContent = mod.name;
        row.replaceChild(name, input);
        syncProgress();
      });

      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') input.blur();
      });

      row.replaceChild(input, name);
      input.focus();
    });

    row.appendChild(checkbox);
    row.appendChild(name);
    moduleList.appendChild(row);
  });
}




    // Calculate overall progress across all subjects.
    // We weight by number of chapters per subject so the overall reflects the
    // proportion of completed chapters across the entire study plan.
    function updateOverall() {
      try {
        let totalChaptersAll = 0;
        let completedChaptersAll = 0;
        subjects.forEach(s => {
          const total = Number(s.totalChapters) || 0;
          const completed = Number(s.completedChapters) || 0;
          totalChaptersAll += total;
          completedChaptersAll += completed;
        });
        let overall = 0;
        if (totalChaptersAll > 0) {
          overall = Math.round((completedChaptersAll / totalChaptersAll) * 100);
        } else if (subjects.length) {
          // Fallback: average individual subject progress if chapter counts are missing
          const total = subjects.reduce((acc, s) => acc + (s.progress || 0), 0);
          overall = Math.round(total / subjects.length);
        }
        overall = clamp(overall, 0, 100);
        overallFill.style.width = `${overall}%`;
        overallPercentEl.textContent = `${overall}%`;
        setMotivation(overall);
      } catch (err) {
        console.warn('Failed to compute overall progress', err);
        overallFill.style.width = `0%`;
        overallPercentEl.textContent = `0%`;
        setMotivation(0);
      }
    }

    // Motivational messages based on progress
    function setMotivation(overall) {
      let msg = '';
      if (overall < 40) {
        msg = "You've started — aim for consistent small wins this week.";
      } else if (overall < 70) {
        msg = "Good progress! Keep a steady rhythm and tackle weak topics.";
      } else if (overall < 90) {
        msg = "Great job! Focus on revisions and mock tests to improve further.";
      } else {
        msg = "Excellent! You're almost there — fine tune and revise.";
      }
      motivationEl.textContent = msg;
    }

    // Render quiz history table with Remove buttons
    function renderQuizzes() {
      quizTableBody.innerHTML = '';
      quizzes.slice().reverse().forEach(q => {
        const tr = document.createElement('tr');
        const status = q.score >= 75 ? 'good' : q.score >= 50 ? 'ok' : 'bad';
        tr.innerHTML = `
          <td>${escapeHtml(q.quizName)}</td>
          <td class="topic-cell"></td>
          <td>${q.score}%</td>
          <td>${prettyDate(q.date)}</td>
          <td class="${status === 'good' ? 'status-good' : status === 'ok' ? 'status-ok' : 'status-bad'}">
            ${status === 'good' ? 'Excellent' : status === 'ok' ? 'Needs Practice' : 'Weak'}
          </td>
          <td><button class="remove-quiz small danger">Remove</button></td>
        `;
        const removeBtn = tr.querySelector('.remove-quiz');
        if (removeBtn) {
          removeBtn.addEventListener('click', async () => {
            if (confirm(`Remove quiz "${q.quizName}"?`)) {
              await deleteQuiz(q.quizName);
            }
          });
        }
        // Populate topic cell with a selectable subject dropdown so user can assign subject
        const topicCell = tr.querySelector('.topic-cell');
        const select = document.createElement('select');
        const emptyOpt = document.createElement('option');
        emptyOpt.value = '';
        emptyOpt.textContent = '-- Select subject --';
        select.appendChild(emptyOpt);
        subjects.forEach(s => {
          const o = document.createElement('option');
          o.value = s.name;
          o.textContent = s.name;
          select.appendChild(o);
        });
        // Preselect if topic matches
        if (q.topic) select.value = q.topic;
        if (select) {
          select.addEventListener('change', () => {
            q.topic = select.value;
            // Persist change locally immediately
            try {
              const raw = localStorage.getItem('progressData');
              let payload = { subjects: [], quizzes: [] };
              if (raw) {
                try { payload = JSON.parse(raw); } catch (e) { payload = { subjects: [], quizzes: [] }; }
              }
              payload.quizzes = payload.quizzes || [];
              const idx = payload.quizzes.findIndex(x => x.quizName === q.quizName);
              if (idx >= 0) payload.quizzes[idx].topic = q.topic; else payload.quizzes.push(q);
              localStorage.setItem('progressData', JSON.stringify(payload));
            } catch (e) { console.warn('Failed to persist quiz topic locally', e); }
            // Try to sync to server (best-effort)
            syncProgress();
            renderQuizzes();
          });
        }
        topicCell.appendChild(select);
        quizTableBody.appendChild(tr);
      });
      // Render weak topics as well
      const weak = quizzes
        .filter(q => q.score < 60)
        .sort((a,b) => a.score - b.score)
        .map(q => ({topic: q.topic, score: q.score}));
      renderWeakTopics(weak);
    }

    function renderWeakTopics(weakArr) {
      weakTopicsEl.innerHTML = '';
      if (!weakArr.length) {
        weakTopicsEl.innerHTML = `<li>No weak topics detected — keep it up!</li>`;
        return;
      }
      weakArr.forEach(w => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${escapeHtml(w.topic)}</span><strong>${w.score}%</strong>`;
        weakTopicsEl.appendChild(li);
      });
    }

    // Compute study streak from activity dates
    function computeStreak(dates) {
      if (!dates || !dates.length) return {current:0, longest:0};

      const set = Array.from(new Set(dates)).sort().reverse();
      const dayNums = set.map(d => dateToDayNum(d));
      const todayNum = dateToDayNum(getYMD(new Date()));
      let current = 0;
      for (let i=0; i<dayNums.length; i++){
        if (dayNums[i] === todayNum - i) current++;
        else break;
      }
      let longest = 1, run = 1;
      for (let i=1;i<dayNums.length;i++){
        if (dayNums[i] === dayNums[i-1] - 1) {
          run++;
          longest = Math.max(longest, run);
        } else run = 1;
      }
      if (dayNums.length===0) longest=0;
      return {current, longest};
    }

    function dateToDayNum(ymd) {
      return Math.floor(new Date(ymd + 'T00:00:00').getTime() / (24*3600*1000));
    }
    function getYMD(d){ return d.toISOString().slice(0,10); }

    // Countdown to target date
    function renderGoalCountdown(targetDate) {
      if (!targetDate) {
        countdownEl.innerHTML = 'Days left: <strong>—</strong>';
        goalTextEl.textContent = 'No goal set.';
        return;
      }
      const today = new Date(getYMD(new Date()) + 'T00:00:00');
      const target = new Date(targetDate + 'T00:00:00');
      const diff = Math.ceil((target - today)/ (1000*3600*24));
      countdownEl.innerHTML = `Days left: <strong>${diff >= 0 ? diff : 0}</strong>`;
      goalTextEl.textContent = `Target date: ${prettyDate(targetDate)} (${diff >=0 ? diff : 0} days left)`;
    }

    function getTipsForTopic(topic) {
      return `Tip for "${topic}": Focused revision and practice quizzes will help strengthen this area.`;
    }

    // Backend sync functions

    async function fetchProgress() {
      try {
        const res = await fetch('/api/user-progress', {credentials: 'include'});
        // Treat 200 with logged_in:false as 'no session' (fallback to local). Avoid noisy 401 handling.
        if (!res.ok) {
          console.warn('Failed to fetch progress, status', res.status);
          // on network error, try local
          loadLocalProgress();
          return;
        }
        const data = await res.json();
        if (data && data.logged_in === false) {
          console.info('Server reports no active session; using local progress stored in the browser.');
          updateSyncStatus('local');
          loadLocalProgress();
          return;
        }
        subjects = data.subjects || [];
        quizzes = data.quizzes || [];
        // Always merge local progressData into memory so recent local quiz results show up
        try {
          mergeLocalIntoMemory();
        } catch (e) {
          console.warn('Failed to merge local progressData', e);
        }
        // Update activityDates from quizzes dates for streak
        activityDates.splice(0, activityDates.length, ...(quizzes.map(q => q.date)));
        renderSubjects();
        updateOverall();
        renderQuizzes();
        const {current: curStreak, longest} = computeStreak(activityDates);
        streakEl.innerHTML = `Streak: <strong>${curStreak} 🔥</strong>`;
        streakTextEl.textContent = `${curStreak} day(s) in a row • Best: ${longest}`;
        renderGoalCountdown(userProfile.targetDate);
      } catch (e) {
        console.error('Failed to fetch progress', e);
      }
    }

    // Merge `localStorage.progressData` into the in-memory `subjects` and `quizzes` arrays.
    // Local entries win on conflicts (by quizName for quizzes, by name for subjects).
    function mergeLocalIntoMemory() {
      const raw = localStorage.getItem('progressData');
      if (!raw) return;
      let local;
      try { local = JSON.parse(raw); } catch (e) { return; }
      // Merge subjects: add any local subjects not present
      (local.subjects || []).forEach(ls => {
        if (!subjects.find(s => s.name === ls.name)) subjects.push(ls);
      });
      // Merge quizzes by quizName (local wins and overwrites server copy)
      (local.quizzes || []).forEach(lq => {
        if (!lq || !lq.quizName) return;
        const idx = quizzes.findIndex(sq => sq.quizName === lq.quizName);
        if (idx >= 0) {
          quizzes[idx] = lq; // local wins
        } else {
          quizzes.push(lq);
        }
      });
      // If we had any local data, mark status local (until sync confirms)
      if ((local.subjects && local.subjects.length) || (local.quizzes && local.quizzes.length)) {
        updateSyncStatus('local');
      }
    }

    async function syncProgress() {
      try {
        const payload = { subjects, quizzes };
        const res = await fetch('/api/sync-progress', {
          method: 'POST',
          credentials: 'include',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          console.warn('Failed to sync progress, server returned', res.status);
          // On network or server error, fallback to local save
          saveLocalProgress();
          updateSyncStatus('local');
          return;
        }
        const resp = await res.json();
        if (resp && resp.logged_in === false) {
          // Server explicitly indicates no session — keep local
          saveLocalProgress();
          updateSyncStatus('local');
          console.info('Server accepted sync but reports no session; data saved locally.');
          return;
        }
        // success
        updateSyncStatus('synced');
        // clear local stored progress if it matches current payload
        try {
          const raw = localStorage.getItem('progressData');
          if (raw) {
            const parsed = JSON.parse(raw);
            if (JSON.stringify(parsed) === JSON.stringify({subjects, quizzes})) {
              localStorage.removeItem('progressData');
            }
          }
        } catch (e) { /* ignore */ }
      } catch (e) {
        console.error('Sync failed', e);
        // Fallback to local save when network errors happen
        saveLocalProgress();
        updateSyncStatus('local');
        console.info('Network error saving progress — data stored locally.');
      }
    }

    // Try to upload any locally stored progress to server when online
    async function attemptAutoSyncFromLocal() {
      const raw = localStorage.getItem('progressData');
      if (!raw) return false;
      if (!navigator.onLine) {
        updateSyncStatus('offline');
        return false;
      }
      try {
        const parsed = JSON.parse(raw);
        // Try to post to server
        const res = await fetch('/api/sync-progress', {
          method: 'POST',
          credentials: 'include',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(parsed),
        });
        if (res.ok) {
          const resp = await res.json();
          if (resp && resp.logged_in === false) {
            updateSyncStatus('local');
            console.info('Server reports no session — keeping progress local');
            return false;
          }
          localStorage.removeItem('progressData');
          subjects = parsed.subjects || [];
          quizzes = parsed.quizzes || [];
          renderSubjects(); updateOverall(); renderQuizzes();
          updateSyncStatus('synced');
          console.info('Local progress synced to server');
          return true;
        } else {
          if (res.status === 401) {
            updateSyncStatus('local');
            console.info('Server requires login — keeping progress local');
            return false;
          }
          updateSyncStatus('error');
          console.warn('Auto sync to server failed with status', res.status);
          return false;
        }
      } catch (e) {
        console.error('Auto sync failed', e);
        updateSyncStatus('error');
        return false;
      }
    }

    function updateSyncStatus(state) {
      if (!syncStatusEl) return;
      let html = 'Sync: <strong>unknown</strong>';
      switch(state) {
        case 'synced': html = 'Sync: <strong>synced</strong>'; break;
        case 'local': html = 'Sync: <strong>local only</strong>'; break;
        case 'offline': html = 'Sync: <strong>offline</strong>'; break;
        case 'error': html = 'Sync: <strong>error</strong>'; break;
        default: html = 'Sync: <strong>unknown</strong>';
      }
      syncStatusEl.innerHTML = html;
    }

    // When the browser regains connectivity, attempt to upload saved progress
    window.addEventListener('online', () => {
      updateSyncStatus('online');
      attemptAutoSyncFromLocal();
    });

    async function deleteQuiz(quizName) {
      try {
        const res = await fetch(`/api/delete-quiz/${encodeURIComponent(quizName)}`, {
          method: 'DELETE',
          credentials: 'include'
        });
        if (res.ok) {
          quizzes = quizzes.filter(q => q.quizName !== quizName);
          // Update activity dates for streak
          activityDates.splice(0, activityDates.length, ...(quizzes.map(q => q.date)));
          renderQuizzes();
          const {current: curStreak, longest} = computeStreak(activityDates);
          streakEl.innerHTML = `Streak: <strong>${curStreak} 🔥</strong>`;
          streakTextEl.textContent = `${curStreak} day(s) in a row • Best: ${longest}`;
          syncProgress();
        } else {
          if (res.status === 401) {
            // Server requires login — perform delete locally and save
            quizzes = quizzes.filter(q => q.quizName !== quizName);
            activityDates.splice(0, activityDates.length, ...(quizzes.map(q => q.date)));
            renderQuizzes();
            const {current: curStreak, longest} = computeStreak(activityDates);
            streakEl.innerHTML = `Streak: <strong>${curStreak} 🔥</strong>`;
            streakTextEl.textContent = `${curStreak} day(s) in a row • Best: ${longest}`;
            saveLocalProgress();
            alert('Quiz deleted locally. Server sync disabled until you log in elsewhere.');
            return;
          }
          alert('Failed to delete quiz');
        }
      } catch (e) {
        console.error('Delete quiz failed', e);
        // On network failure, delete locally and persist
        quizzes = quizzes.filter(q => q.quizName !== quizName);
        activityDates.splice(0, activityDates.length, ...(quizzes.map(q => q.date)));
        renderQuizzes();
        const {current: curStreak, longest} = computeStreak(activityDates);
        streakEl.innerHTML = `Streak: <strong>${curStreak} 🔥</strong>`;
        streakTextEl.textContent = `${curStreak} day(s) in a row • Best: ${longest}`;
        saveLocalProgress();
      }
    }

    // Simple login flow: prompt username and call backend /login to create a session
    // Previously a login prompt; we intentionally avoid asking for login.
    // Instead we use localStorage to persist progress when server sync is unavailable.
    function loginFlow() {
      return false;
    }

    // Local persistence helpers
    function saveLocalProgress() {
      try {
        const payload = { subjects, quizzes };
        localStorage.setItem('progressData', JSON.stringify(payload));
      } catch (e) {
        console.error('Failed to save local progress', e);
      }
    }

    function loadLocalProgress() {
      try {
        const raw = localStorage.getItem('progressData');
        if (!raw) {
          subjects = [];
          quizzes = [];
          renderSubjects();
          renderQuizzes();
          updateOverall();
          return false;
        }
        const data = JSON.parse(raw);
        subjects = data.subjects || [];
        quizzes = data.quizzes || [];
        activityDates.splice(0, activityDates.length, ...(quizzes.map(q => q.date)));
        renderSubjects();
        updateOverall();
        renderQuizzes();
        const {current: curStreak, longest} = computeStreak(activityDates);
        streakEl.innerHTML = `Streak: <strong>${curStreak} 🔥</strong>`;
        streakTextEl.textContent = `${curStreak} day(s) in a row • Best: ${longest}`;
        renderGoalCountdown(userProfile.targetDate);
        return true;
      } catch (e) {
        console.error('Failed to load local progress', e);
        return false;
      }
    }
const addSubjectForm = document.getElementById('add-subject-form');
const newSubjectNameInput = document.getElementById('newSubjectName');
const newSubjectModulesInput = document.getElementById('newSubjectModules');

addSubjectForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const name = newSubjectNameInput.value.trim();
  let modules = parseInt(newSubjectModulesInput.value, 10);
  if (!name) {
    alert("Please enter a valid subject name");
    return;
  }
  if (isNaN(modules) || modules < 0) modules = 0;

  subjects.push({
    name,
    totalChapters: modules,
    completedChapters: 0,
    progress: 0,
    modules: []
  });

  renderSubjects();
  updateOverall();
  syncProgress();

  addSubjectForm.reset();
});


    // Save button syncs current progress
    if (btnSave) {
  btnSave.addEventListener('click', () => {
    syncProgress();
    alert('Progress synced to backend.');

    // Collapse all expanded module dropdowns
    const expandedModules = document.querySelectorAll('.module-list-panel:not(.hidden)');
    expandedModules.forEach(panel => {
      panel.classList.add('hidden'); // hide module list panel
      // Also update the corresponding expand button arrow to 'down' (▼)
      const subjectItem = panel.closest('.subject-item');
      if (subjectItem) {
        const expandBtn = subjectItem.querySelector('.expand-btn');
        if (expandBtn) expandBtn.innerHTML = "&#9660;";
      }
    });
  });
}


    // Get tips button shows tip for first weak topic
    if (btnGetTips) {
      btnGetTips.addEventListener('click', () => {
        const weakNodes = weakTopicsEl.querySelectorAll('li');
        if (!weakNodes.length || weakNodes[0].textContent.includes('No weak')) {
          alert('No weak topics found — keep practising!');
          return;
        }
        const first = weakNodes[0].querySelector('span').textContent;
        alert(getTipsForTopic(first));
      });
    }

    // Initial load from backend
    fetchProgress();

    // Expose state for debugging
    window.progressTracker = { get subjects() { return subjects; }, get quizzes() { return quizzes; }, fetchProgress, syncProgress };
  });
