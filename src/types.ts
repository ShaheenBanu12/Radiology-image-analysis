import { RadiomicsFeatures, MLModelPrediction } from './services/classicalML';

export interface ScanAnalysis {
  id: string;
  timestamp: string;
  imageType: 'CT' | 'MRI' | 'PET/CT';
  status: 'completed' | 'pending' | 'error';
  doctorReport: string;
  patientSummary: string;
  confidence: number;
  region: string;
  abnormalityDetected: boolean;
  imageUrl?: string;
  patientName?: string;
  radiomicsFeatures?: RadiomicsFeatures;
  randomForestResult?: MLModelPrediction;
  xgboostResult?: MLModelPrediction;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export type ViewState = 'dashboard' | 'analyze';
