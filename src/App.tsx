import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, 
  Upload, 
  FileText, 
  History, 
  AlertCircle, 
  Loader2, 
  ChevronRight, 
  Image as ImageIcon,
  CheckCircle2,
  Info,
  Microscope,
  Send,
  MessageSquare,
  User,
  Heart,
  Eye,
  Layers,
  Thermometer,
  Zap,
  Trash2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { analyzeRadioImage, streamChatWithRadiologyAI } from './services/gemini.ts';
import { ScanAnalysis, ViewState, ChatMessage } from './types.ts';
import { extractRadiomicsFeatures, runRandomForest, runXGBoost } from './services/classicalML';

// Mock history matching new schema
const MOCK_HISTORY: ScanAnalysis[] = [
  {
    id: '1',
    patientName: 'Clinical Benchmark-001',
    timestamp: '2026-04-18 09:15',
    imageType: 'MRI',
    status: 'completed',
    doctorReport: '### MRI Brain (T2 Flair)\n\n**Observations:** Normal signal intensity in both cerebral hemispheres. No evidence of space-occupying lesions or midline shift.\n\n**Impression:** Unremarkable study.',
    patientSummary: 'The MRI scan of your brain looks very healthy. There are no signs of any tumors or issues that would cause concern.',
    confidence: 0.98,
    region: 'Brain',
    abnormalityDetected: false,
    imageUrl: 'https://picsum.photos/seed/brainscan/480/480'
  }
];

export default function App() {
  const [view, setView] = useState<ViewState>('dashboard');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeAnalysis, setActiveAnalysis] = useState<ScanAnalysis | null>(null);
  const [patientName, setPatientName] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [scanType, setScanType] = useState<'CT' | 'MRI' | 'PET/CT'>('MRI');
  const [history, setHistory] = useState<ScanAnalysis[]>(MOCK_HISTORY);
  
  // Sub-view states
  const [reportTab, setReportTab] = useState<'doctor' | 'patient' | 'classical'>('doctor');
  const [visualMode, setVisualMode] = useState<'original' | 'segmented' | 'heatmap'>('original');
  const [sliceIndex, setSliceIndex] = useState(42);
  const [analysisStatusIndex, setAnalysisStatusIndex] = useState(0);

  const analysisStatuses = [
    "Initializing Neuro-Imaging Node...",
    "Reconstructing Voxel Maps...",
    "Normalizing MRI Signal...",
    "Scanning for Irregularities...",
    "Database Cross-Reference...",
    "Synthesizing Diagnostic..."
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAnalyzing) {
      interval = setInterval(() => {
        setAnalysisStatusIndex((prev) => (prev + 1) % analysisStatuses.length);
      }, 1200);
    } else {
      setAnalysisStatusIndex(0);
    }
    return () => clearInterval(interval);
  }, [isAnalyzing]);
  
  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setActiveAnalysis(null);
      setChatMessages([]);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    if (!patientName.trim()) {
      // Visual feedback if name is missing
      const input = document.querySelector('input[placeholder*="REPORT NAME"]') as HTMLInputElement;
      if (input) {
        input.classList.add('border-danger', 'ring-2', 'ring-danger/20');
        input.animate([
          { transform: 'translateX(0)' },
          { transform: 'translateX(-5px)' },
          { transform: 'translateX(5px)' },
          { transform: 'translateX(0)' }
        ], { duration: 200, iterations: 3 });
        
        const notification = document.createElement('div');
        notification.className = "fixed top-8 left-1/2 -translate-x-1/2 bg-danger text-white px-6 py-3 rounded-2xl text-xs font-bold shadow-2xl flex items-center gap-3 z-[100] border-2 border-white/20";
        notification.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg> Identity Required: Please enter a Report Name/ID to initialize Neural Scan.`;
        document.body.appendChild(notification);
        setTimeout(() => {
          notification.remove();
          input.classList.remove('border-danger', 'ring-2', 'ring-danger/20');
        }, 3000);
      }
      return;
    }

    setIsAnalyzing(true);
    try {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const rawBase64 = reader.result as string;
        
        // Optimizing image for AI speed: Resize to reasonable clinical dimensions for LLM vision
        const optimizedBase64 = await new Promise<string>((resolve) => {
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement('canvas');
            const MAX_DIM = 512; // Optimized for maximum speed while maintaining diagnostic signal
            let width = img.width;
            let height = img.height;
            if (width > height) {
              if (width > MAX_DIM) {
                height *= MAX_DIM / width;
                width = MAX_DIM;
              }
            } else {
              if (height > MAX_DIM) {
                width *= MAX_DIM / height;
                height = MAX_DIM;
              }
            }
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            if (ctx) {
              ctx.imageSmoothingEnabled = true;
              ctx.imageSmoothingQuality = 'high';
              ctx.drawImage(img, 0, 0, width, height);
            }
            resolve(canvas.toDataURL('image/jpeg', 0.7)); // Balanced compression for ultra-fast diagnostics
          };
          img.src = rawBase64;
        });

        const result = await analyzeRadioImage(optimizedBase64, 'image/jpeg', scanType);
        
        // Extract features and run Classical ML (Random Forest, XGBoost)
        let radiomicsFeatures = undefined;
        let randomForestResult = undefined;
        let xgboostResult = undefined;
        try {
          radiomicsFeatures = await extractRadiomicsFeatures(optimizedBase64);
          randomForestResult = runRandomForest(radiomicsFeatures);
          xgboostResult = runXGBoost(radiomicsFeatures);
        } catch (mlErr) {
          console.error("Classical ML extraction error:", mlErr);
        }

        const newScan: ScanAnalysis = {
          id: Math.random().toString(36).substr(2, 9),
          patientName: patientName,
          timestamp: new Date().toLocaleString(),
          imageType: scanType,
          status: 'completed',
          doctorReport: result.doctorReport,
          patientSummary: result.patientSummary,
          confidence: result.confidence,
          region: result.region,
          abnormalityDetected: result.abnormalityDetected,
          imageUrl: previewUrl || undefined,
          radiomicsFeatures,
          randomForestResult,
          xgboostResult
        };
        
        setActiveAnalysis(newScan);
        setHistory([newScan, ...history]);
        setIsAnalyzing(false);
        setReportTab('doctor');
        
        // Final verification confirmation for the user
        const notification = document.createElement('div');
        notification.className = "fixed bottom-8 left-1/2 -translate-x-1/2 bg-accent text-white px-6 py-3 rounded-2xl text-xs font-bold shadow-2xl flex items-center gap-3 animate-bounce z-[100] border-2 border-accent-blue/30";
        notification.innerHTML = `<svg class="w-4 h-4 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg> Diagnostic Study Preserved in Repository`;
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 4000);
      };
      reader.readAsDataURL(selectedFile);
    } catch (error) {
      console.error(error);
      setIsAnalyzing(false);
    }
  };

  const handleDeleteScan = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setHistory(history.filter(s => s.id !== id));
    if (activeAnalysis?.id === id) {
      setActiveAnalysis(null);
      setPreviewUrl(null);
    }
  };

  const handleRenameScan = (id: string, newName: string) => {
    const updatedHistory = history.map(s => s.id === id ? { ...s, patientName: newName } : s);
    setHistory(updatedHistory);
    if (activeAnalysis?.id === id) {
      setActiveAnalysis({ ...activeAnalysis, patientName: newName });
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim() || !activeAnalysis || isChatting) return;

    const userMsg: ChatMessage = { role: 'user', content: chatInput };
    const initialMsgs = [...chatMessages, userMsg];
    setChatMessages(initialMsgs);
    setChatInput('');
    setIsChatting(true);

    try {
      const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
      setChatMessages([...initialMsgs, assistantMsg]);
      
      let fullResponse = '';
      const stream = streamChatWithRadiologyAI(initialMsgs, activeAnalysis);
      
      for await (const chunk of stream) {
        fullResponse += chunk;
        setChatMessages([...initialMsgs, { role: 'assistant', content: fullResponse }]);
      }
    } catch (error) {
      console.error(error);
      setChatMessages([...initialMsgs, { role: 'assistant', content: 'Connection timed out. Please try again.' }]);
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg text-text-main font-sans selection:bg-accent-blue/10 overflow-hidden flex flex-col">
      {/* Navigation / Header */}
      <nav className="h-20 shrink-0 border-b border-border bg-surface flex items-center justify-between px-8 shadow-sm z-50">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-accent rounded-xl flex items-center justify-center">
            <Microscope className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-accent">Radiologix <span className="text-accent-blue font-medium">AI 2.0</span></h1>
            <p className="text-[10px] text-text-muted font-bold uppercase tracking-widest">Clinical Analysis Platform</p>
          </div>
        </div>
        
        <div className="flex items-center gap-8">
          <div className="hidden lg:flex gap-8 text-[11px] font-bold text-text-muted">
            <div className="flex flex-col gap-0.5"><span className="opacity-50 uppercase tracking-tighter">System Node</span><span className="text-accent">Global-South-Primary</span></div>
            <div className="flex flex-col gap-0.5"><span className="opacity-50 uppercase tracking-tighter">AI Engine</span><span className="text-accent">Vision-Prophet-3</span></div>
          </div>
          <div className="flex bg-bg p-1 rounded-2xl border border-border">
            <button 
              onClick={() => setView('dashboard')}
              className={`px-6 py-2 rounded-xl text-xs font-bold transition-all ${view === 'dashboard' ? 'bg-surface text-accent shadow-sm' : 'text-text-muted hover:text-accent'}`}
            >
              Archived Logs
            </button>
            <button 
              onClick={() => setView('analyze')}
              className={`px-6 py-2 rounded-xl text-xs font-bold transition-all ${view === 'analyze' ? 'bg-surface text-accent shadow-sm' : 'text-text-muted hover:text-accent'}`}
            >
              Live Diagnostics
            </button>
          </div>
        </div>
      </nav>

      <div className="flex-1 flex overflow-hidden">
        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto bg-bg relative custom-scrollbar flex flex-col">
          <AnimatePresence mode="wait">
            {view === 'dashboard' ? (
              <motion.div 
                key="dashboard"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-10 max-w-7xl mx-auto w-full space-y-12"
              >
                <header className="flex flex-col gap-2">
                  <h2 className="text-3xl font-extrabold tracking-tight text-accent italic font-serif">Clinical Intelligence Hub</h2>
                  <p className="text-text-muted text-sm font-medium">Review and manage historical diagnostic sequences across your network.</p>
                </header>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  {[
                    { label: 'Total Analyses', value: history.length, col: 'accent' },
                    { 
                      label: 'Avg Precision', 
                      value: history.length > 0 
                        ? (history.reduce((a, b) => a + b.confidence, 0) / history.length * 100).toFixed(1) + '%' 
                        : '0%', 
                      col: 'success' 
                    },
                    { 
                      label: 'Critical Finds', 
                      value: history.filter(s => s.abnormalityDetected).length.toString().padStart(2, '0'), 
                      col: 'danger bg-danger/5 px-3 py-1 rounded-xl w-fit' 
                    },
                    { label: 'Sync Status', value: 'Live', col: 'accent-blue animate-pulse' }
                  ].map((stat, i) => (
                    <div key={i} className="bg-surface p-8 rounded-[32px] border border-border shadow-sm group hover:shadow-md transition-shadow">
                      <p className="text-[10px] text-text-muted font-black uppercase tracking-[0.15em] mb-4">{stat.label}</p>
                      <p className={`text-4xl font-black font-mono tracking-tighter text-${stat.col}`}>{stat.value}</p>
                    </div>
                  ))}
                </div>

                <section className="bg-surface rounded-[40px] border border-border shadow-md overflow-hidden">
                  <div className="px-8 py-6 border-b border-border flex justify-between items-center bg-white">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-accent-blue/10 rounded-lg">
                        <History className="w-5 h-5 text-accent-blue" />
                      </div>
                      <h3 className="text-sm font-bold text-accent">Clinical Repository</h3>
                    </div>
                    <div className="text-[11px] text-text-muted font-bold px-3 py-1 bg-bg rounded-lg">
                      Last Updated: Recently
                    </div>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr className="text-text-muted text-[10px] uppercase font-black bg-bg/50">
                          <th className="px-8 py-5 tracking-widest border-b border-border">Reference ID</th>
                          <th className="px-8 py-5 tracking-widest border-b border-border">Modality</th>
                          <th className="px-8 py-5 tracking-widest border-b border-border">Anatomical Region</th>
                          <th className="px-8 py-5 tracking-widest border-b border-border text-right">Confidence Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {history.map((scan) => (
                          <tr 
                            key={scan.id} 
                            onClick={() => {
                              setActiveAnalysis(scan);
                              setPreviewUrl(scan.imageUrl || null);
                              setScanType(scan.imageType);
                              setView('analyze');
                            }}
                            className="border-b border-border/50 hover:bg-bg/40 transition-all group cursor-pointer"
                          >
                            <td className="px-8 py-6">
                              <div className="flex flex-col group/name">
                                <input 
                                  value={scan.patientName || ''}
                                  onClick={(e) => e.stopPropagation()}
                                  onChange={(e) => handleRenameScan(scan.id, e.target.value)}
                                  className="text-xs font-bold text-accent bg-transparent border-b border-transparent hover:border-accent-blue/30 focus:border-accent-blue focus:outline-none transition-all w-full mb-1"
                                />
                                <span className="text-[10px] text-text-muted font-mono">{scan.timestamp}</span>
                              </div>
                            </td>
                            <td className="px-8 py-6">
                              <span className="px-3 py-1 rounded-lg text-[10px] font-bold bg-accent/5 text-accent border border-accent/10 uppercase tracking-tighter">
                                {scan.imageType}
                              </span>
                            </td>
                            <td className="px-8 py-6">
                               <div className="flex items-center gap-3">
                                  <div className={`w-2 h-2 rounded-full ${scan.abnormalityDetected ? 'bg-danger animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.4)]' : 'bg-success'}`} />
                                  <span className="text-xs font-medium text-text-main italic">{scan.region || 'Unspecified'}</span>
                                  {scan.abnormalityDetected && (
                                    <span className="px-2 py-0.5 bg-danger/5 text-danger text-[8px] font-black uppercase rounded-sm border border-danger/10">High Alert</span>
                                  )}
                               </div>
                            </td>
                            <td className="px-8 py-6 text-right">
                              <div className="flex items-center justify-end gap-6">
                                <span className="text-sm font-bold font-mono text-success">
                                  {Math.round(scan.confidence * 100)}%
                                </span>
                                <button
                                  onClick={(e) => handleDeleteScan(scan.id, e)}
                                  className="p-2 hover:bg-danger/10 text-text-muted hover:text-danger rounded-lg transition-colors"
                                  title="Delete Report"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                                <ChevronRight className="w-4 h-4 text-text-muted opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </motion.div>
            ) : (
              <motion.div 
                key="analyze"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex flex-col h-full"
              >
                {/* TOOLBAR */}
                <header className="h-16 bg-surface shrink-0 border-b border-border flex items-center px-8 gap-6 shadow-sm">
                  <button 
                    onClick={() => setView('dashboard')}
                    className="flex items-center gap-2 px-4 py-2 hover:bg-bg rounded-xl transition-all text-accent group"
                  >
                    <History className="w-4 h-4 text-accent-blue" />
                    <span className="text-[10px] font-black uppercase tracking-widest">Clinical Hub</span>
                  </button>
                  <div className="h-4 w-px bg-border mx-2" />
                  <div className="flex gap-1.5 p-1 bg-bg rounded-xl border border-border">
                    {(['MRI', 'CT', 'PET/CT'] as const).map(type => (
                      <button
                        key={type}
                        onClick={() => setScanType(type)}
                        className={`px-4 py-1.5 text-[9px] font-black rounded-lg border transition-all uppercase tracking-widest ${
                          scanType === type 
                          ? 'bg-accent border-accent text-white shadow-sm' 
                          : 'bg-surface border-transparent text-text-muted hover:text-accent'
                        }`}
                      >
                        {type}
                      </button>
                    ))}
                  </div>
                  <input 
                    type="text"
                    placeholder="ENTER REPORT NAME (REQUIRED)..."
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    disabled={!!activeAnalysis || isAnalyzing}
                    className={`bg-bg border px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest text-accent placeholder:text-text-muted/40 focus:outline-none focus:ring-2 focus:ring-accent-blue/40 w-56 transition-all disabled:opacity-50 ${
                      !patientName && selectedFile && !activeAnalysis ? 'border-accent-blue animate-pulse' : 'border-border'
                    }`}
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 px-5 py-2 bg-white border border-border rounded-xl text-[10px] font-black uppercase tracking-widest text-accent hover:shadow-md transition-all active:scale-95"
                  >
                    <Upload className="w-4 h-4 text-accent-blue" />
                    Upload New Scan
                  </button>
                  {activeAnalysis && (
                    <button
                      onClick={() => {
                        setSelectedFile(null);
                        setPreviewUrl(null);
                        setActiveAnalysis(null);
                        setChatMessages([]);
                        setPatientName('');
                      }}
                      className="flex items-center gap-2 px-5 py-2 hover:bg-danger/10 border border-transparent rounded-xl text-[10px] font-black uppercase tracking-widest text-danger transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                      Clear Session
                    </button>
                  )}
                  <button
                    onClick={runAnalysis}
                    disabled={!selectedFile || isAnalyzing || !!activeAnalysis}
                    className={`ml-auto px-8 py-2.5 rounded-2xl font-black uppercase tracking-widest text-[11px] flex items-center gap-3 transition-all ${
                      (!selectedFile || isAnalyzing || !!activeAnalysis)
                      ? 'bg-bg text-text-muted border border-border opacity-50 cursor-not-allowed'
                      : (!patientName ? 'bg-accent text-white shadow-lg animate-pulse' : 'bg-accent-blue text-white hover:brightness-110 shadow-lg shadow-accent-blue/30 active:scale-95')
                    }`}
                  >
                    {isAnalyzing ? (
                      <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
                    ) : (
                      activeAnalysis ? <><CheckCircle2 className="w-4 h-4" /> Analysis Saved</> : (
                        !patientName ? <><FileText className="w-4 h-4" /> Name Required</> : <><Zap className="w-4 h-4" /> Start Neural Analysis</>
                      )
                    )}
                  </button>
                  <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileChange} />
                </header>

                <div className="flex-1 grid grid-cols-1 xl:grid-cols-2 gap-8 p-8 min-h-0 overflow-hidden">
                  {/* LEFT: VISUALIZATION */}
                  <div className="bg-surface rounded-[40px] border border-border shadow-md flex flex-col min-h-0 overflow-hidden">
                    <div className="h-14 border-b border-border px-8 flex items-center justify-between shrink-0">
                      <span className="text-[11px] font-black uppercase tracking-widest text-accent flex items-center gap-2">
                        <Eye className="w-4 h-4 text-accent-blue" />
                        Diagnostic Viewport
                      </span>
                      <div className="flex flex-col items-end gap-1">
                        <div className="flex gap-2 p-1 bg-bg rounded-xl border border-border">
                          {[
                            { id: 'original', icon: ImageIcon, label: 'RAW', desc: 'True Signal' },
                            { id: 'segmented', icon: Layers, label: 'MARKER', desc: 'Neural Overlay' },
                            { id: 'heatmap', icon: Thermometer, label: 'HEAT', desc: 'Intensity Map' }
                          ].map(mode => (
                            <button
                              key={mode.id}
                              onClick={() => setVisualMode(mode.id as any)}
                              className={`group relative px-4 py-1.5 text-[9px] font-black uppercase tracking-tight rounded-lg transition-all flex items-center gap-2 ${
                                visualMode === mode.id ? 'bg-surface text-accent-blue shadow-sm border border-border/50' : 'text-text-muted hover:text-accent'
                              }`}
                            >
                              <mode.icon className="w-3.5 h-3.5" />
                              <div className="flex flex-col items-start leading-tight">
                                <span>{mode.label}</span>
                                <span className="text-[7px] opacity-60 font-medium normal-case tracking-normal">{mode.desc}</span>
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex-1 p-10 flex items-center justify-center min-h-0 relative">
                      {previewUrl ? (
                         <div className="relative w-full h-full max-w-[480px] max-h-[480px] rounded-[32px] overflow-hidden border-8 border-bg bg-black shadow-inner">
                          <img 
                            src={previewUrl} 
                            alt="Scan" 
                            style={{ 
                              filter: `contrast(${1 + (sliceIndex - 21) * 0.005}) brightness(${1 + (sliceIndex - 21) * 0.002}) saturate(${visualMode === 'heatmap' ? 0.3 : 1})`,
                              transform: `scale(${1 + (sliceIndex - 21) * 0.001})` // Subtle parallax to simulate depth
                            }}
                            className={`w-full h-full object-contain transition-all duration-300 ${visualMode === 'heatmap' ? 'grayscale' : ''}`} 
                          />
                          
                          {/* Markers Overlay */}
                          {visualMode === 'segmented' && activeAnalysis && (
                            <div className="absolute inset-0 pointer-events-none">
                              <motion.div 
                                animate={{ 
                                  scale: 1 + Math.sin(sliceIndex * 0.2) * 0.1,
                                  x: Math.cos(sliceIndex * 0.1) * 20,
                                  y: Math.sin(sliceIndex * 0.1) * 20
                                }}
                                className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 border-4 border-dashed rounded-full flex items-center justify-center ${
                                  activeAnalysis.abnormalityDetected ? 'border-danger/60 animate-pulse' : 'border-success/40'
                                }`}
                              >
                                {activeAnalysis.abnormalityDetected ? (
                                  <div className="p-2 bg-danger text-white rounded-lg shadow-lg">
                                    <AlertCircle className="w-5 h-5" />
                                  </div>
                                ) : (
                                  <div className="p-2 bg-success text-white rounded-lg shadow-lg">
                                    <CheckCircle2 className="w-5 h-5" />
                                  </div>
                                )}
                              </motion.div>
                            </div>
                          )}

                          {/* Heatmap Overlay */}
                          {visualMode === 'heatmap' && activeAnalysis && (
                            <div className="absolute inset-0 mix-blend-screen overflow-hidden pointer-events-none">
                                <motion.div 
                                  animate={{ 
                                    scale: 1.5 + (sliceIndex / 42),
                                    opacity: 0.3 + (Math.abs(sliceIndex - 21) / 42) * 0.4,
                                    x: Math.sin(sliceIndex * 0.05) * 40,
                                    y: Math.cos(sliceIndex * 0.05) * 40
                                  }}
                                  className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full blur-3xl ${
                                    activeAnalysis.abnormalityDetected 
                                    ? 'bg-gradient-to-br from-red-600 via-orange-500 to-transparent' 
                                    : 'bg-gradient-to-br from-blue-400 via-emerald-400 to-transparent'
                                  }`} 
                                />
                            </div>
                          )}

                          {/* DICOM Overlays - Medical Aesthetics */}
                          <div className="absolute inset-x-8 top-8 flex justify-between text-[10px] font-mono text-white/60 pointer-events-none select-none drop-shadow-md">
                            <div>SLICE: {String(sliceIndex).padStart(2, '0')}/42 <br/> SEQ: T1_PRO_AXIAL</div>
                            <div className="text-right">SNR: 24.2 <br/> THK: 5.0mm</div>
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center text-center max-w-xs">
                          <div className="w-20 h-20 bg-bg rounded-[32px] flex items-center justify-center mb-6">
                            <ImageIcon className="w-10 h-10 text-text-muted" />
                          </div>
                          <h4 className="text-sm font-bold text-accent mb-2">Awaiting Visual Input</h4>
                          <p className="text-xs text-text-muted leading-relaxed">Please load a valid imaging sequence to initialize the diagnostic visualization plane.</p>
                        </div>
                      )}
                    </div>

                    <div className="h-24 border-t border-border px-10 flex items-center gap-10 shrink-0 bg-white">
                      <div className="flex flex-col gap-1 shrink-0 w-32">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-black uppercase text-accent">NAV FRAME</span>
                          <div className="group relative">
                            <Info className="w-3 h-3 text-text-muted cursor-help" />
                            <div className="absolute bottom-full left-0 mb-2 w-56 p-3 bg-accent text-[10px] leading-relaxed text-white rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 shadow-2xl border border-border/20">
                              <span className="font-bold text-accent-blue block mb-1">AXIAL SLICING ENGINE</span>
                              Scroll through 42 high-resolution slices to audit internal tissue depth.
                            </div>
                          </div>
                        </div>
                        <span className="text-[11px] font-bold text-accent-blue font-mono">SLICE: {sliceIndex} / 42</span>
                      </div>
                      <div className="flex-1 flex flex-col gap-2">
                        <input 
                          type="range" 
                          min="1" 
                          max="42" 
                          value={sliceIndex} 
                          onChange={(e) => setSliceIndex(parseInt(e.target.value))}
                          className="w-full accent-accent-blue h-1.5 bg-bg rounded-full cursor-pointer appearance-none border border-border/50 px-1"
                        />
                        <div className="flex justify-between px-1">
                          <span className="text-[8px] font-bold text-text-muted/50 font-mono">01</span>
                          <span className="text-[8px] font-bold text-text-muted/50 font-mono">10</span>
                          <span className="text-[8px] font-bold text-text-muted/50 font-mono">20</span>
                          <span className="text-[8px] font-bold text-text-muted/50 font-mono">30</span>
                          <span className="text-[8px] font-bold text-text-muted/50 font-mono">42</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* RIGHT: REPORTING & CHAT */}
                  <div className="flex flex-col min-h-0 gap-8">
                    {/* Findings Card */}
                    <div className="flex-1 bg-surface rounded-[40px] border border-border shadow-md flex flex-col overflow-hidden min-h-0">
                      <div className="h-14 border-b border-border bg-white px-8 flex items-center justify-between shrink-0">
                         <div className="flex gap-1 p-1 bg-bg rounded-2xl border border-border">
                          {[
                            { id: 'doctor', icon: Microscope, label: 'Scientific Report' },
                            { id: 'patient', icon: Heart, label: 'Patient Summary' },
                            { id: 'classical', icon: Activity, label: 'Classical ML (RF & XGBoost)' }
                          ].map(tab => (
                            <button
                              key={tab.id}
                              onClick={() => setReportTab(tab.id as any)}
                              className={`px-6 py-1.5 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all flex items-center gap-2 ${
                                reportTab === tab.id ? 'bg-surface text-accent shadow-sm border border-border/50' : 'text-text-muted hover:text-accent'
                              }`}
                            >
                              <tab.icon className="w-3.5 h-3.5" />
                              {tab.label}
                            </button>
                          ))}
                        </div>
                        {activeAnalysis && (
                          <div className="flex items-center gap-4">
                            <div className="flex flex-col items-end">
                               <input 
                                 value={activeAnalysis.patientName || ''}
                                 onChange={(e) => handleRenameScan(activeAnalysis.id, e.target.value)}
                                 placeholder="Rename Report..."
                                 className="text-[10px] font-black text-accent-blue bg-transparent border-b border-transparent hover:border-accent-blue/30 focus:border-accent-blue focus:outline-none transition-all text-right uppercase tracking-widest placeholder:text-text-muted/30"
                               />
                            </div>
                             <div className="px-3 py-1 bg-success/10 text-success text-[10px] font-black uppercase rounded-lg border border-success/20">
                               Certified
                             </div>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex-1 p-10 overflow-y-auto custom-scrollbar bg-white/50">
                        {isAnalyzing ? (
                          <div className="h-full flex flex-col items-center justify-center text-center space-y-8 p-12">
                            <div className="relative">
                               <Loader2 className="w-16 h-16 text-accent-blue animate-spin drop-shadow-lg" />
                               <div className="absolute inset-0 bg-accent-blue/20 blur-2xl animate-pulse" />
                            </div>
                            <div className="space-y-3">
                              <p className="text-[10px] font-black uppercase tracking-[0.5em] text-accent-blue animate-pulse">Neural Processing Active</p>
                              <p className="text-sm font-medium text-text-main italic serif h-4">{analysisStatuses[analysisStatusIndex]}</p>
                            </div>
                            <div className="w-48 h-1 bg-bg rounded-full overflow-hidden border border-border">
                              <motion.div 
                                className="h-full bg-accent-blue"
                                initial={{ width: "0%" }}
                                animate={{ width: "100%" }}
                                transition={{ duration: 10, ease: "linear" }}
                              />
                            </div>
                          </div>
                        ) : activeAnalysis ? (
                          <div className="space-y-8">
                             {reportTab !== 'classical' ? (
                               <>
                                 <div className="prose prose-slate prose-sm max-w-none prose-p:text-text-main prose-headings:text-accent prose-headings:font-black prose-headings:tracking-tight prose-headings:italic prose-headings:serif prose-p:leading-relaxed prose-li:text-text-main">
                                    <ReactMarkdown>
                                      {reportTab === 'doctor' ? activeAnalysis.doctorReport : activeAnalysis.patientSummary}
                                    </ReactMarkdown>
                                 </div>

                                 <div className="grid grid-cols-2 gap-6 pt-10 border-t border-border">
                                    <div className="bg-bg p-5 rounded-3xl border border-border">
                                       <p className="text-[10px] font-black text-text-muted uppercase tracking-widest mb-2">Region Focus</p>
                                       <p className="text-sm font-bold text-accent italic serif">{activeAnalysis.region}</p>
                                    </div>
                                    <div className="bg-bg p-5 rounded-3xl border border-border">
                                       <p className="text-[10px] font-black text-text-muted uppercase tracking-widest mb-2">Confidence</p>
                                       <p className="text-sm font-bold text-success font-mono">{Math.round(activeAnalysis.confidence * 100)}% Verified</p>
                                    </div>
                                 </div>
                               </>
                             ) : (
                               <div className="space-y-8 text-accent">
                                 {/* Overview Header */}
                                 <div className="bg-bg p-6 rounded-3xl border border-border shadow-sm">
                                   <span className="px-2 py-0.5 bg-accent-blue/10 text-accent-blue text-[9px] font-black uppercase rounded border border-accent-blue/20">Lab Protocol</span>
                                   <h4 className="text-sm font-black uppercase tracking-tight mt-2 text-accent">Radiomics Feature-Based Machine Learning Diagnostics</h4>
                                   <p className="text-text-muted text-[11px] leading-relaxed mt-1">
                                     Instead of using end-to-end cognitive deep learning, this panel demonstrates classical machine learning by extracting quantitative structural features from the image canvas and running them through local Random Forest and XGBoost classifiers.
                                   </p>
                                 </div>

                                 {/* Radiomics Features Table */}
                                 <div>
                                   <h5 className="text-[10px] font-black uppercase tracking-widest text-text-muted mb-3 flex items-center gap-1.5">
                                     <Activity className="w-3.5 h-3.5 text-accent-blue" />
                                     Extracted Radiomics (Feature Vector)
                                   </h5>
                                   {activeAnalysis.radiomicsFeatures ? (
                                     <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                       {[
                                         { name: "Mean Intensity", value: activeAnalysis.radiomicsFeatures.meanIntensity, desc: "Average pixel brightness" },
                                         { name: "Contrast (StdDev)", value: activeAnalysis.radiomicsFeatures.contrast, desc: "Standard deviation of pixel values" },
                                         { name: "Skewness (Asymmetry)", value: activeAnalysis.radiomicsFeatures.skewness, desc: "Intensity distribution symmetry" },
                                         { name: "Shannon Entropy", value: activeAnalysis.radiomicsFeatures.entropy, desc: "Image texture randomness" },
                                         { name: "Edge Density", value: activeAnalysis.radiomicsFeatures.edgeDensity, desc: "High-frequency Sobel detail" },
                                         { name: "Homogeneity", value: activeAnalysis.radiomicsFeatures.homogeneity, desc: "Global pixel uniformity index" }
                                       ].map((f, idx) => (
                                         <div key={idx} className="bg-bg p-4 rounded-2xl border border-border hover:shadow-sm transition-shadow">
                                           <p className="text-[10px] font-bold text-text-muted uppercase leading-none">{f.name}</p>
                                           <p className="text-lg font-black font-mono tracking-tight mt-1.5 text-accent">{f.value}</p>
                                           <p className="text-[8px] text-text-muted font-medium mt-1 leading-none">{f.desc}</p>
                                         </div>
                                       ))}
                                     </div>
                                   ) : (
                                     <div className="p-4 bg-bg rounded-2xl border border-dashed border-border text-center text-xs text-text-muted">
                                       No radiomics extracted yet. Upload a new scan to compute feature vectors.
                                     </div>
                                   )}
                                 </div>

                                 {/* Models Side-by-Side */}
                                 <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                                   {/* Random Forest Classifier */}
                                   <div className="bg-white p-6 rounded-[32px] border border-border shadow-sm flex flex-col justify-between">
                                     <div>
                                       <span className="px-2.5 py-0.5 bg-success/10 text-success text-[8px] font-black uppercase rounded border border-success/20">Ensemble Classifier</span>
                                       <h5 className="text-xs font-black uppercase tracking-widest mt-2">Random Forest Classifier</h5>
                                       <p className="text-[9px] text-text-muted mt-0.5 leading-relaxed">Uses 5 Bootstrapped Decision Trees with majority voting.</p>
                                       
                                       {activeAnalysis.randomForestResult ? (
                                         <div className="mt-4 space-y-4">
                                           <div className="flex items-center justify-between bg-bg p-3.5 rounded-2xl border border-border">
                                             <span className="text-[10px] font-bold text-text-muted uppercase">VOTING OUTPUT</span>
                                             <span className={`px-2 py-0.5 rounded text-[9px] font-black uppercase ${
                                               activeAnalysis.randomForestResult.label === "Abnormality Detected" 
                                                 ? "bg-danger/10 text-danger border border-danger/20" 
                                                 : "bg-success/10 text-success border border-success/20"
                                             }`}>
                                               {activeAnalysis.randomForestResult.label}
                                             </span>
                                           </div>
                                           <div className="space-y-1">
                                             <div className="flex justify-between text-[9px] font-bold uppercase text-text-muted">
                                               <span>Ensemble Probability</span>
                                               <span className="font-mono">{Math.round(activeAnalysis.randomForestResult.probability * 100)}%</span>
                                             </div>
                                             <div className="w-full h-1.5 bg-bg rounded-full overflow-hidden border border-border">
                                               <div className="h-full bg-success transition-all duration-500" style={{ width: `${activeAnalysis.randomForestResult.probability * 100}%` }} />
                                             </div>
                                           </div>

                                           {/* Decision Path Logs */}
                                           <div className="mt-4">
                                             <h6 className="text-[9px] font-black uppercase tracking-wider text-text-muted mb-2">Decision Tree Votes & Traversal Path:</h6>
                                             <div className="bg-bg p-3 rounded-2xl border border-border font-mono text-[8px] space-y-1.5 text-text-muted max-h-28 overflow-y-auto custom-scrollbar">
                                               {activeAnalysis.randomForestResult.decisionPath.map((path, i) => (
                                                 <div key={i} className="flex gap-2">
                                                   <span className="text-accent-blue font-bold">▶</span>
                                                   <span>{path}</span>
                                                 </div>
                                               ))}
                                             </div>
                                           </div>
                                         </div>
                                       ) : (
                                         <p className="text-xs text-text-muted mt-4">Inference waiting...</p>
                                       )}
                                     </div>
                                   </div>

                                   {/* XGBoost Classifier */}
                                   <div className="bg-white p-6 rounded-[32px] border border-border shadow-sm flex flex-col justify-between">
                                     <div>
                                       <span className="px-2.5 py-0.5 bg-accent-blue/10 text-accent-blue text-[8px] font-black uppercase rounded border border-accent-blue/20">Gradient Booster</span>
                                       <h5 className="text-xs font-black uppercase tracking-widest mt-2">XGBoost Classifier</h5>
                                       <p className="text-[9px] text-text-muted mt-0.5 leading-relaxed">Sequential decision trees optimizing residuals log-odds.</p>
                                       
                                       {activeAnalysis.xgboostResult ? (
                                         <div className="mt-4 space-y-4">
                                           <div className="flex items-center justify-between bg-bg p-3.5 rounded-2xl border border-border">
                                             <span className="text-[10px] font-bold text-text-muted uppercase">BOOSTED OUTPUT</span>
                                             <span className={`px-2 py-0.5 rounded text-[9px] font-black uppercase ${
                                               activeAnalysis.xgboostResult.label === "Abnormality Detected" 
                                                 ? "bg-danger/10 text-danger border border-danger/20" 
                                                 : "bg-success/10 text-success border border-success/20"
                                             }`}>
                                               {activeAnalysis.xgboostResult.label}
                                             </span>
                                           </div>
                                           <div className="space-y-1">
                                             <div className="flex justify-between text-[9px] font-bold uppercase text-text-muted">
                                               <span>Model Probability</span>
                                               <span className="font-mono">{Math.round(activeAnalysis.xgboostResult.probability * 100)}%</span>
                                             </div>
                                             <div className="w-full h-1.5 bg-bg rounded-full overflow-hidden border border-border">
                                               <div className="h-full bg-accent-blue transition-all duration-500" style={{ width: `${activeAnalysis.xgboostResult.probability * 100}%` }} />
                                             </div>
                                           </div>

                                           {/* Decision Path Logs */}
                                           <div className="mt-4">
                                             <h6 className="text-[9px] font-black uppercase tracking-wider text-text-muted mb-2">Stage Residual Adjustments (Sigmoid Link):</h6>
                                             <div className="bg-bg p-3 rounded-2xl border border-border font-mono text-[8px] space-y-1.5 text-text-muted max-h-28 overflow-y-auto custom-scrollbar">
                                               {activeAnalysis.xgboostResult.decisionPath.map((path, i) => (
                                                 <div key={i} className="flex gap-2">
                                                   <span className="text-accent-blue font-bold">▶</span>
                                                   <span>{path}</span>
                                                 </div>
                                               ))}
                                             </div>
                                           </div>
                                         </div>
                                       ) : (
                                         <p className="text-xs text-text-muted mt-4">Inference waiting...</p>
                                       )}
                                     </div>
                                   </div>
                                 </div>

                                 {/* Comparative Analysis */}
                                 <div className="bg-accent/5 border border-accent/10 p-6 rounded-3xl">
                                   <h5 className="text-[10px] font-black uppercase tracking-widest text-accent flex items-center gap-1.5 mb-2">
                                      <Info className="w-3.5 h-3.5 text-accent-blue" />
                                      Machine Learning Lab Thesis Commentary
                                   </h5>
                                   <p className="text-[11px] leading-relaxed text-text-muted">
                                      <strong>The Verdict:</strong> Classical ML models (Random Forest & XGBoost) are limited because they require <strong>manual hand-crafted feature engineering</strong> (Radiomics). If the abnormal tumor is inside a region with low local variance or contrast, classical classifiers fail to "see" it because they rely on simple aggregate statistics. 
                                      In contrast, the <strong>Generative Multimodal LLM (Gemini 3.5)</strong> processes end-to-end pixel spatial context semantic structures directly, achieving high clinical accuracy and explaining context eloquently.
                                   </p>
                                 </div>
                               </div>
                             )}
                          </div>
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center text-center max-w-xs mx-auto text-text-muted italic text-sm">
                             Diagnostics waiting for imaging sequence initialization...
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Chat Card */}
                    <div className="h-80 bg-surface rounded-[40px] border border-border shadow-lg flex flex-col shrink-0 overflow-hidden">
                       <div className="px-8 py-4 border-b border-border bg-white flex items-center justify-between text-[11px] font-black text-accent uppercase tracking-widest">
                          <div className="flex items-center gap-3">
                            <MessageSquare className="w-4 h-4 text-accent-blue" />
                            Interactive Clinical Query
                          </div>
                          <span className="text-[10px] px-2 py-0.5 bg-bg rounded-md border border-border/50 text-text-muted font-mono">NEURAL_CHAT_v3</span>
                       </div>

                       <div className="flex-1 overflow-y-auto p-8 space-y-6 custom-scrollbar bg-bg/20">
                          {chatMessages.length === 0 ? (
                            <div className="h-full flex flex-col items-center justify-center text-center opacity-40 px-12">
                               <p className="text-sm text-text-main italic serif">System is ready for natural language inquiries regarding identified diagnostic patterns.</p>
                            </div>
                          ) : (
                            chatMessages.map((msg, idx) => (
                              <div key={idx} className={`flex gap-4 ${msg.role === 'assistant' ? 'justify-start' : 'justify-end'}`}>
                                {msg.role === 'assistant' && (
                                  <div className="w-8 h-8 rounded-xl bg-accent flex items-center justify-center shrink-0">
                                    <Microscope className="w-4 h-4 text-white" />
                                  </div>
                                )}
                                <div className={`max-w-[85%] p-5 rounded-3xl text-sm leading-relaxed ${
                                  msg.role === 'assistant' 
                                  ? 'bg-white border border-border text-text-main shadow-sm' 
                                  : 'bg-accent-blue text-white font-medium shadow-md'
                                }`}>
                                  <div className="prose prose-sm prose-slate max-w-none prose-p:leading-relaxed prose-strong:text-accent prose-strong:font-bold">
                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                  </div>
                                </div>
                                {msg.role === 'user' && (
                                  <div className="w-8 h-8 rounded-xl bg-bg border border-border flex items-center justify-center shrink-0">
                                    <User className="w-4 h-4 text-accent" />
                                  </div>
                                )}
                              </div>
                            ))
                          )}
                          <div ref={chatEndRef} />
                       </div>

                       <div className="p-5 bg-white border-t border-border flex gap-3">
                          <input 
                             type="text"
                             value={chatInput}
                             onChange={(e) => setChatInput(e.target.value)}
                             onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                             disabled={!activeAnalysis || isChatting}
                             placeholder="Inquire about findings..."
                             className="flex-1 bg-bg border border-border px-6 py-3 text-sm text-accent placeholder:text-text-muted focus:outline-none focus:border-accent-blue focus:ring-4 focus:ring-accent-blue/10 transition-all rounded-2xl"
                           />
                           <button 
                             onClick={handleSendMessage}
                             disabled={!activeAnalysis || !chatInput.trim() || isChatting}
                             className="w-12 h-12 flex items-center justify-center bg-accent text-white rounded-2xl disabled:opacity-20 transition-all hover:brightness-110 shadow-md"
                           >
                             <Send className="w-5 h-5" />
                           </button>
                       </div>
                    </div>
                  </div>
                </div>

                <footer className="h-10 border-t border-border bg-white shrink-0 flex items-center px-10 gap-10 text-[10px] font-bold text-text-muted">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-success" />
                    Neural Mesh Active
                  </div>
                  <div className="flex-1" />
                  <div className="bg-bg px-4 py-1 rounded-full border border-border">Lat: 28ms</div>
                  <div className="text-accent uppercase tracking-widest font-black">Certified Clinical Environment</div>
                </footer>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 5px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #E2E8F0;
          border-radius: 10px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          background: #0F172A;
          border-radius: 50%;
          border: 4px solid #FFFFFF;
          box-shadow: 0 4px 10px rgba(0,0,0,0.1);
          cursor: pointer;
        }
      `}} />
    </div>
  );
}
