import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  getDocs,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  where,
  writeBatch,
} from "firebase/firestore";
import { db, isConfigured } from "../firebase/config";

const requireDb = () => {
  if (!isConfigured || !db) {
    throw new Error(
      "Firebase is not configured. Check the VITE_FIREBASE_* values in .env.",
    );
  }
};

/**
 * Firestore does not accept undefined values.
 * This helper removes undefined recursively while keeping
 * strings, numbers, booleans, arrays and nested objects intact.
 */
function removeUndefined(value) {
  if (Array.isArray(value)) {
    return value.filter((item) => item !== undefined).map(removeUndefined);
  }

  if (value && typeof value === "object" && !(value instanceof Date)) {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .map(([key, item]) => [key, removeUndefined(item)]),
    );
  }

  return value;
}

export async function createConversation(userId, title = "New Query") {
  requireDb();

  const ref = await addDoc(collection(db, "conversations"), {
    userId,
    title: title.trim() || "New Query",
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  });

  return ref.id;
}

export async function updateConversationTitle(conversationId, title) {
  requireDb();

  const safeTitle = String(title || "New Query").trim() || "New Query";

  await updateDoc(doc(db, "conversations", conversationId), {
    title: safeTitle,
    updatedAt: serverTimestamp(),
  });
}

export async function getUserConversations(userId) {
  requireDb();

  const conversationsQuery = query(
    collection(db, "conversations"),
    where("userId", "==", userId),
  );

  const snapshot = await getDocs(conversationsQuery);

  return snapshot.docs
    .map((item) => ({
      id: item.id,
      ...item.data(),
    }))
    .sort((a, b) => {
      const aTime = a.updatedAt?.toMillis?.() || a.createdAt?.toMillis?.() || 0;
      const bTime = b.updatedAt?.toMillis?.() || b.createdAt?.toMillis?.() || 0;
      return bTime - aTime;
    });
}

export async function addMessage(conversationId, message) {
  requireDb();

  const safeMessage = removeUndefined(message);

  const messagesRef = collection(
    db,
    "conversations",
    conversationId,
    "messages",
  );

  const messageRef = await addDoc(messagesRef, {
    ...safeMessage,
    createdAt: serverTimestamp(),
  });

  await updateDoc(doc(db, "conversations", conversationId), {
    updatedAt: serverTimestamp(),
  });

  return messageRef.id;
}

export async function replaceMessage(conversationId, messageId, message) {
  requireDb();

  const safeMessage = removeUndefined(message);
  const messageRef = doc(
    db,
    "conversations",
    conversationId,
    "messages",
    messageId,
  );

  await setDoc(messageRef, {
    ...safeMessage,
    createdAt: serverTimestamp(),
  });

  await updateDoc(doc(db, "conversations", conversationId), {
    updatedAt: serverTimestamp(),
  });
}

export async function getConversationMessages(conversationId) {
  requireDb();

  const messagesQuery = query(
    collection(db, "conversations", conversationId, "messages"),
    orderBy("createdAt", "asc"),
  );

  const snapshot = await getDocs(messagesQuery);

  return snapshot.docs.map((item) => ({
    id: item.id,
    ...item.data(),
  }));
}

/**
 * Firestore write batches are limited to 500 operations.
 * Delete messages in safe chunks, then delete the conversation.
 */
export async function deleteConversation(conversationId) {
  requireDb();

  const messagesSnapshot = await getDocs(
    collection(db, "conversations", conversationId, "messages"),
  );

  const refs = messagesSnapshot.docs.map((message) => message.ref);
  const CHUNK_SIZE = 450;

  for (let index = 0; index < refs.length; index += CHUNK_SIZE) {
    const chunk = refs.slice(index, index + CHUNK_SIZE);
    const batch = writeBatch(db);

    chunk.forEach((messageRef) => batch.delete(messageRef));
    await batch.commit();
  }

  await deleteDoc(doc(db, "conversations", conversationId));
}

/**
 * Builds the Firestore-safe assistant payload used to reconstruct a saved RAG answer.
 * Undefined values are removed by addMessage/removeUndefined before the write.
 */
export function buildAssistantMessageData(response = {}) {
  return {
    role: "assistant",
    text: response.text ?? "",
    title: response.title,
    confidence: response.confidence,
    sources: response.sources,
    pipelineVersion: response.pipelineVersion,
    answerValid: response.answerValid,
    retrievalConfidence: response.retrievalConfidence,
    answerConfidence: response.answerConfidence,
    evidenceGraph: response.evidenceGraph,
    evidenceState: response.evidenceState,
    agreement: response.agreement,
    complexity: response.complexity,
    margin: response.margin,
    stability: response.stability,
    recommendedTopK: response.recommendedTopK,
  };
}
