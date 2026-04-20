import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export async function analyzeRadioImage(imageData: string, mimeType: string, scanType: string) {
  const model = "gemini-3-flash-preview"; 
  
  const analysisSchema = {
    type: Type.OBJECT,
    properties: {
      doctorReport: { type: Type.STRING, description: "Technical report" },
      patientSummary: { type: Type.STRING, description: "Simple summary" },
      confidence: { type: Type.NUMBER, description: "Confidence 0-1" },
      region: { type: Type.STRING, description: "Anatomical region" },
      abnormalityDetected: { type: Type.BOOLEAN, description: "Finding present" }
    },
    required: ["doctorReport", "patientSummary", "confidence", "region", "abnormalityDetected"]
  };

  try {
    const response = await ai.models.generateContent({
      model,
      contents: [
        {
          role: "user",
          parts: [
            { text: `Quick clinical audit: ${scanType} image. Identify findings and summarize.` },
            { inlineData: { data: imageData.split(',')[1], mimeType } }
          ]
        }
      ],
      config: {
        systemInstruction: `You are an expert Board-Certified Radiologist performing a high-precision clinical audit. 
        
        YOUR MISSION:
        1. Objectivity: Maintain strict clinical neutrality. If the ${scanType} image shows normal anatomical structures with no pathology, you MUST mark 'abnormalityDetected' as FALSE. 
        2. Accuracy: Do not hallucinate findings. If the scan is a "Normal Study", describe it as such in the report.
        3. Logic: Only flag abnormalities if there are clear visual indicators of lesions, masses, fractures, inflammation, or structural irregularities.
        
        Output must be a high-speed JSON audit. Concise but precise. No fluff.`,
        temperature: 0.1,
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        responseMimeType: "application/json",
        responseSchema: analysisSchema
      }
    });

    const text = response.text;
    if (!text) throw new Error("Empty response");
    return JSON.parse(text);
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    throw new Error("Diagnostic node connection error.");
  }
}

export async function* streamChatWithRadiologyAI(messages: {role: string, content: string}[], currentCase: any) {
  const model = "gemini-3-flash-preview";
  
  const moodPrompt = currentCase.abnormalityDetected 
    ? "The scan shows findings that require medical attention. Maintain a deeply supportive, empathetic, and serious tone. Use words that convey professional concern and care (human-like 'sadness' for the news but strength in support)."
    : "The scan results are normal. Maintain a warm, encouraging, and relieved tone. Express professional happiness for the positive outcome.";

  const systemInstruction = `You are a world-class AI Radiologist and Clinical Communicator, designed to match the conversational depth of Claude or Gemini.

  ${moodPrompt}

  CONTEXT OF CURRENT ANALYSIS:
  - Technical Insights: ${currentCase.doctorReport}
  - Simple Summary: ${currentCase.patientSummary}
  - Anatomical Area: ${currentCase.region}
  
  STRICT INTERACTION RULES:
  1. RESPONSE STRUCTURE: Always use Markdown. Use bullet points for lists. Bold key clinical terms.
  2. SCOPE: Discuss ONLY the findings from this specific ${currentCase.imageType} scan. 
  3. EMPATHY: Acknowledge the user's feelings first if they express anxiety or joy.
  4. PRECISION: Be direct. Do not waffle. If something is healthy, say it clearly. If something is suspicious, explain why simply.
  5. DISCLAIMER: End briefly by stating this is research support and to verify with their personal physician.

  Maintain a high-IQ, emotionally intelligent persona throughout.`;

  const contents = messages.map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }]
  }));

  try {
    const response = await ai.models.generateContentStream({
      model,
      contents,
      config: { 
        systemInstruction,
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW }
      }
    });

    for await (const chunk of response) {
      if (chunk.text) yield chunk.text;
    }
  } catch (error) {
    console.error("Chat Error:", error);
    yield "I'm having trouble connecting to the analysis brain. Please try again.";
  }
}
