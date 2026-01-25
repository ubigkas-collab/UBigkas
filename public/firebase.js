// firebase.js
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";
import { getStorage } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-storage.js";


const firebaseConfig = {
  apiKey: "AIzaSyC7woa1g1tRQKoepK1m_CQfkyjfwqMcOmc",
  authDomain: "ubigkas-2226b.firebaseapp.com",
  projectId: "ubigkas-2226b",
  storageBucket: "ubigkas-2226b.appspot.com",
  messagingSenderId: "63784933292",
  appId: "1:63784933292:web:fe7ba2b306d429ad5b4c09"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);
