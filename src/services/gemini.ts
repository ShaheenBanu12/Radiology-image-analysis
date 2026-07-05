export async function analyzeRadioImage(
  imageData: string, 
  mimeType: string, 
  scanType: string,
  age?: number,
  gender?: string,
  cholesterol?: number
) {
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ imageData, mimeType, scanType, age, gender, cholesterol }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || "Diagnostic node connection error.");
    }

    return await response.json();
  } catch (error: any) {
    console.error("Analysis client error:", error);
    throw error;
  }
}

export async function* streamChatWithRadiologyAI(messages: {role: string, content: string}[], currentCase: any) {
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages, currentCase }),
    });

    if (!response.ok) {
      yield "I'm having trouble connecting to the analysis brain. Please try again.";
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      yield "Streaming error: Reader not available.";
      return;
    }

    const decoder = new TextDecoder("utf-8");
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  } catch (error) {
    console.error("Chat client error:", error);
    yield "I'm having trouble connecting to the analysis brain. Please try again.";
  }
}
