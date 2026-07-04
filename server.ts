import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "50mb" })); // Support large base64 medical images

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    }
  }
});

// API endpoint for image analysis
app.post("/api/analyze", async (req, res) => {
  const { imageData, mimeType, scanType } = req.body;
  if (!imageData || !mimeType || !scanType) {
    return res.status(400).json({ error: "Missing required parameters" });
  }

  const model = "gemini-3.5-flash"; 
  
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
    const base64Data = imageData.includes(",") ? imageData.split(",")[1] : imageData;
    const response = await ai.models.generateContent({
      model,
      contents: [
        {
          role: "user",
          parts: [
            { text: `Quick clinical audit: ${scanType} image. Identify findings and summarize.` },
            { inlineData: { data: base64Data, mimeType } }
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
        responseMimeType: "application/json",
        responseSchema: analysisSchema
      }
    });

    const text = response.text;
    if (!text) throw new Error("Empty response");
    res.json(JSON.parse(text));
  } catch (error: any) {
    console.error("Gemini Analysis Error:", error);
    res.status(500).json({ error: "Diagnostic node connection error. " + error.message });
  }
});

// API endpoint for streaming chat
app.post("/api/chat", async (req, res) => {
  const { messages, currentCase } = req.body;
  if (!messages || !currentCase) {
    return res.status(400).json({ error: "Missing required parameters" });
  }

  const model = "gemini-3.5-flash";
  
  const moodPrompt = currentCase.abnormalityDetected 
    ? "The scan shows findings that require medical attention. Maintain a deeply supportive, empathetic, and serious tone. Use words that convey professional concern and care."
    : "The scan results are normal. Maintain a warm, encouraging, and relieved tone. Express professional happiness for the positive outcome.";

  const systemInstruction = `You are a world-class AI Radiologist and Clinical Communicator.

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

  const contents = messages.map((m: any) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }]
  }));

  try {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const stream = await ai.models.generateContentStream({
      model,
      contents,
      config: { 
        systemInstruction
      }
    });

    for await (const chunk of stream) {
      if (chunk.text) {
        res.write(chunk.text);
      }
    }
    res.end();
  } catch (error: any) {
    console.error("Chat Error:", error);
    res.write("I'm having trouble connecting to the analysis brain. Please try again.");
    res.end();
  }
});

// Vite server integration or static file serve
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
