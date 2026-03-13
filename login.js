// ---------------- SIGN UP ----------------
const signupForm = document.getElementById('signupForm');
if (signupForm) {
    signupForm.addEventListener('submit', async function(e){
        e.preventDefault();

        const name = document.getElementById('signupName').value.trim();
        const email = document.getElementById('signupEmail').value.trim();
        const password = document.getElementById('signupPassword').value.trim();

        if(!name || !email || !password){
            // NOTE: alert() is used here as per the original code, but 
            // a custom modal should be used in production.
            alert("Please fill all fields!");
            return;
        }
      
        if(password.length < 6){
            alert("Password should be at least 6 characters long!");
            return;
        }

        try {
            // ✅ V8/Compat: Create user in Firebase Authentication
            const userCredential = await auth.createUserWithEmailAndPassword(email, password);
            const user = userCredential.user;

            // ✅ V8/Compat: Save extra user data in Firestore
            await db.collection("users").doc(user.uid).set({
                name: name,
                email: email,
                createdAt: firebase.firestore.FieldValue.serverTimestamp() // Best practice for timestamp
            });

            alert("✅ Signup successful! User created in Firebase Auth.");
            signupForm.reset();

            // Redirect to profile page
            //toggle to sign-in view
            window.location.href = "profile.html";

        } catch (error) {
            console.error("Signup error:", error);

            if (error.code === "auth/email-already-in-use") {
                alert("⚠️ Email already registered. Please log in instead.");
                // Simulate a click on the sign-in button if it exists
                document.getElementById('signIn')?.click(); 
            } else {
                alert("❌ " + error.message);
            }
            signupForm.reset();
        }
    });
}

//-----SIGN IN-----
const signinForm = document.getElementById("signinForm");
if (signinForm) {
  signinForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const email = document.getElementById("signinEmail").value.trim();
    const password = document.getElementById("signinPassword").value.trim();

    if (!email || !password) {
        alert("Please fill all fields!");
        return;
    }

    try {
      // ✅ V8/Compat: Sign In
      const userCredential = await auth.signInWithEmailAndPassword(email, password);
      alert("✅ Logged in as: " + userCredential.user.email);
      console.log("User object:", userCredential.user);
      signinForm.reset();
      window.location.href = "index.html";
    } catch (error) {
      console.error(error);
      alert("❌ " + error.message);
      signinForm.reset();
    }
  });
}

// --- Overlay panels ---
const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');
const container = document.getElementById('container');
if (signUpButton && signInButton && container) {
    signUpButton.addEventListener('click', () => {
        container.classList.add("right-panel-active");
    });
    signInButton.addEventListener('click', () => {
        container.classList.remove("right-panel-active");
    });
}
