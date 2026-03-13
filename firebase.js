//define firebase and its services
// Firebase v10+ (compat SDKs for non-module usage)

const firebaseConfig = {
  apiKey: "AIzaSyCsdv2DcDa_u-kNBAXuq3TXp9nfdLy-chs",
  authDomain: "margadarshi-85118.firebaseapp.com",
  databaseURL: "https://margadarshi-85118-default-rtdb.firebaseio.com",
  projectId: "margadarshi-85118",
  storageBucket: "margadarshi-85118.firebasestorage.app",
  messagingSenderId: "868205365403",
  appId: "1:868205365403:web:2f2abcc026c0a88968de3e"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Make Firestore/Auth/Storage available globally
window.db = firebase.firestore();
window.auth = firebase.auth();

// Note: For modular SDK usage (ESM), use the following imports instead:
/*
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";
*/
