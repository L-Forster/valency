'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Scale, Tick } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { EnhancedAnalysisResult } from '../../types';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

interface AnalysisResultsProps {
  result: EnhancedAnalysisResult;
  originalText: string;
}

export default function AnalysisResults({ result, originalText }: AnalysisResultsProps) {
  // Prepare data for emotional arc chart - using bar chart instead of curved line
  const chartData = {
    labels: result.emotionalArc.map((_, index) => `Sentence ${index + 1}`),
    datasets: [
      {
        label: 'Valence (Positive/Negative)',
        data: result.emotionalArc.map(point => point.valence),
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.7)',
        borderWidth: 2,
        borderRadius: 4,
      },
      {
        label: 'Arousal (Energy/Intensity)',
        data: result.emotionalArc.map(point => point.arousal),
        borderColor: 'rgb(236, 72, 153)',
        backgroundColor: 'rgba(236, 72, 153, 0.7)',
        borderWidth: 2,
        borderRadius: 4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        min: -1,
        max: 1,
        ticks: {
          callback: function(this: Scale, tickValue: number | string) {
            const value = Number(tickValue);
            if (value === 1) return 'High';
            if (value === 0) return 'Neutral';
            if (value === -1) return 'Low';
            return '';
          }
        },
        grid: {
          color: 'rgba(200, 200, 200, 0.1)'
        }
      },
      x: {
        grid: {
          color: 'rgba(200, 200, 200, 0.1)'
        }
      }
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            let description = '';
            
            if (context.datasetIndex === 0) { // Valence
              if (value > 0.5) description = 'Very Positive';
              else if (value > 0) description = 'Somewhat Positive';
              else if (value > -0.5) description = 'Somewhat Negative';
              else description = 'Very Negative';
            } else { // Arousal
              if (value > 0.5) description = 'High Energy';
              else if (value > 0) description = 'Somewhat Energetic';
              else if (value > -0.5) description = 'Somewhat Calm';
              else description = 'Very Calm';
            }
            
            return `${label}: ${value.toFixed(2)} (${description})`;
          },
        },
      },
    },
  };

  // Function to determine color based on valence and arousal - increased sensitivity for smaller values
  const getEmotionColor = (valence: number, arousal: number) => {
    // Positive valence, high arousal (excited, happy)
    if (valence > 0.2 && arousal > 0.2) {
      return 'bg-yellow-200 text-yellow-800 border-yellow-400 font-medium';
    }
    // Positive valence, low arousal (content, relaxed)
    else if (valence > 0.2 && arousal <= 0.2) {
      return 'bg-green-200 text-green-800 border-green-400 font-medium';
    }
    // Negative valence, high arousal (angry, anxious)
    else if (valence <= -0.2 && arousal > 0.2) {
      return 'bg-red-200 text-red-800 border-red-400 font-medium';
    }
    // Negative valence, low arousal (sad, depressed)
    else if (valence <= -0.2 && arousal <= 0.2) {
      return 'bg-blue-200 text-blue-800 border-blue-400 font-medium';
    }
    // Slightly positive
    else if (valence > 0) {
      return 'bg-green-100 text-green-700 border-green-300';
    }
    // Slightly negative
    else if (valence < 0) {
      return 'bg-red-100 text-red-700 border-red-300';
    }
    // Neutral
    else {
      return 'bg-gray-100 text-gray-700 border-gray-300';
    }
  };

  // Function to identify most important sentences (highest influence)
  const getImportanceClass = (normalizedInfluence: number) => {
    if (normalizedInfluence >= 0.25) {
      return 'bg-indigo-100 border-l-4 border-indigo-500 pl-2';
    }
    return '';
  };

  // Function to render highlighted text
  const renderHighlightedText = () => {
    if (!originalText || !result.wordAnalysis || result.wordAnalysis.length === 0) {
      return <p className="text-gray-700">{originalText}</p>;
    }

    // Sort word analysis by startIndex to ensure proper order
    const sortedWordAnalysis = [...result.wordAnalysis].sort((a, b) => a.startIndex - b.startIndex);
    
    const textParts: React.ReactNode[] = [];
    let lastIndex = 0;

    // Identify sentences with high influence for highlighting
    const highInfluenceSentences = result.sentenceBreakdown
      .filter(s => s.normalizedInfluence >= 0.2)
      .map(s => s.sentence);

    sortedWordAnalysis.forEach((word, index) => {
      // Add text before the highlighted word
      if (word.startIndex > lastIndex) {
        textParts.push(
          <span key={`text-${index}`} className="text-gray-700">
            {originalText.substring(lastIndex, word.startIndex)}
          </span>
        );
      }

      // Add the highlighted word with more prominent colors
      const colorClass = getEmotionColor(word.valence, word.arousal);
      textParts.push(
        <span 
          key={`word-${index}`} 
          className={`px-1 py-0.5 rounded border ${colorClass}`}
          title={`Valence: ${word.valence.toFixed(2)}, Arousal: ${word.arousal.toFixed(2)}`}
        >
          {originalText.substring(word.startIndex, word.endIndex)}
        </span>
      );

      lastIndex = word.endIndex;
    });

    // Add any remaining text
    if (lastIndex < originalText.length) {
      textParts.push(
        <span key="text-end" className="text-gray-700">
          {originalText.substring(lastIndex)}
        </span>
      );
    }

    return <p className="leading-relaxed whitespace-pre-line">{textParts}</p>;
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Analysis Results</h3>
        
        {/* Overall Valence and Arousal */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Overall Valence</h4>
            <div className="flex items-center">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className={`h-2.5 rounded-full ${result.overallEmotion.valence > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                  style={{ width: `${Math.abs(result.overallEmotion.valence * 100)}%` }}
                ></div>
              </div>
              <span className="ml-2 text-lg font-semibold">
                {result.overallEmotion.valence.toFixed(2)}
              </span>
            </div>
            <p className="mt-1 text-sm text-gray-500">
              {result.overallEmotion.valence > 0.5 ? 'Very Positive' : 
               result.overallEmotion.valence > 0 ? 'Somewhat Positive' :
               result.overallEmotion.valence > -0.5 ? 'Somewhat Negative' : 'Very Negative'}
            </p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Overall Arousal</h4>
            <div className="flex items-center">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="h-2.5 rounded-full bg-purple-500"
                  style={{ width: `${Math.abs(result.overallEmotion.arousal * 100)}%` }}
                ></div>
              </div>
              <span className="ml-2 text-lg font-semibold">
                {result.overallEmotion.arousal.toFixed(2)}
              </span>
            </div>
            <p className="mt-1 text-sm text-gray-500">
              {result.overallEmotion.arousal > 0.5 ? 'High Energy' : 
               result.overallEmotion.arousal > 0 ? 'Moderate Energy' :
               result.overallEmotion.arousal > -0.5 ? 'Somewhat Calm' : 'Very Calm'}
            </p>
          </div>
        </div>
        
        {/* Highlighted Conversation */}
        <div className="mb-8">
          <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Highlighted Conversation</h4>
          <div className="bg-gray-50 p-4 rounded-lg">
            {renderHighlightedText()}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <div className="flex items-center">
              <span className="inline-block w-3 h-3 bg-yellow-200 border border-yellow-400 rounded mr-1"></span>
              <span className="text-xs text-gray-500">Excited/Happy</span>
            </div>
            <div className="flex items-center">
              <span className="inline-block w-3 h-3 bg-green-200 border border-green-400 rounded mr-1"></span>
              <span className="text-xs text-gray-500">Content/Relaxed</span>
            </div>
            <div className="flex items-center">
              <span className="inline-block w-3 h-3 bg-red-200 border border-red-400 rounded mr-1"></span>
              <span className="text-xs text-gray-500">Angry/Anxious</span>
            </div>
            <div className="flex items-center">
              <span className="inline-block w-3 h-3 bg-blue-200 border border-blue-400 rounded mr-1"></span>
              <span className="text-xs text-gray-500">Sad/Depressed</span>
            </div>
          </div>
        </div>
        
        {/* Emotional Arc Chart - Changed to Bar chart */}
        <div className="mb-8">
          <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Emotional Arc</h4>
          <div className="bg-gray-50 p-4 rounded-lg" style={{ height: '300px' }}>
            <Bar data={chartData} options={chartOptions} />
          </div>
        </div>
        
        {/* Key Transitions - Updated text colors */}
        {result.key_transitions && result.key_transitions.length > 0 && (
          <div className="mb-8">
            <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Key Emotional Transitions</h4>
            <div className="space-y-2">
              {result.key_transitions.map((transition, index) => (
                <div 
                  key={`transition-${index}`}
                  className={`p-3 rounded-lg ${
                    transition.severity === 'info' ? 'bg-blue-50 border-l-4 border-blue-400' 
                    : transition.severity === 'warning' ? 'bg-amber-50 border-l-4 border-amber-400'
                    : 'bg-red-50 border-l-4 border-red-400'
                  }`}
                >
                  <p className={`text-sm font-medium ${
                    transition.severity === 'info' ? 'text-blue-700' 
                    : transition.severity === 'warning' ? 'text-amber-700'
                    : 'text-red-700'
                  }`}>
                    {transition.description}
                  </p>
                  <p className="text-xs text-gray-500">
                    After Sentence {transition.position}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Improvement Recommendations */}
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Improvement Recommendations</h4>
          <ul className="space-y-2">
            {result.recommendedImprovements.map((recommendation, index) => (
              <li key={`recommendation-${index}`} className="flex items-start">
                <svg className="h-5 w-5 text-indigo-500 mt-0.5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="text-gray-700">{recommendation}</span>
              </li>
            ))}
          </ul>
        </div>
        
        {/* Sentence Breakdown - with highlighting for important sentences */}
        <div>
          <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Sentence Breakdown</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sentence</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Valence</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Arousal</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Influence</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {result.sentenceBreakdown.map((sentence, index) => (
                  <tr key={`sentence-${index}`} className={sentence.normalizedInfluence >= 0.2 ? 'bg-indigo-50' : ''}>
                    <td className="px-6 py-4">
                      <div className={`text-sm text-gray-900 ${getImportanceClass(sentence.normalizedInfluence)}`}>
                        {sentence.sentence}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div 
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          sentence.predictedValence > 0.2 ? 'bg-green-100 text-green-800' : 
                          sentence.predictedValence < -0.2 ? 'bg-red-100 text-red-800' : 
                          'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {sentence.predictedValence.toFixed(2)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div 
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          sentence.predictedArousal > 0.3 ? 'bg-purple-100 text-purple-800' : 
                          sentence.predictedArousal < -0.2 ? 'bg-blue-100 text-blue-800' : 
                          'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {sentence.predictedArousal.toFixed(2)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${sentence.normalizedInfluence >= 0.2 ? 'text-indigo-700' : 'text-gray-900'}`}>
                        {(sentence.normalizedInfluence * 100).toFixed(0)}%
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}