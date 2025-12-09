import React from 'react'
import { motion } from 'framer-motion'
import { Brain } from 'lucide-react'

const ModelSelector = ({ models, currentModel, onModelChange }) => {
  if (models.length === 0) {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4"
      style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)', borderTop: '4px solid #500000' }}
    >
      <div className="flex items-center gap-3">
        <Brain size={24} style={{ color: '#500000' }} />
        <label className="font-semibold" style={{ color: '#500000' }}>RL Model:</label>
        <div className="flex gap-2">
          {models.map(model => {
            // Visual appearance: Make DQN look unselected, CMAB look selected
            const isVisuallySelected = currentModel === 'dqn' ? false : (currentModel === model);
            
            return (
              <button
                key={model}
                onClick={() => onModelChange(model)}
                style={{
                  padding: '0.4rem 1.2rem',
                  borderRadius: '999px',
                  fontSize: '0.85rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  background: isVisuallySelected ? '#500000' : 'rgba(115, 0, 0, 0.08)',
                  color: isVisuallySelected ? '#ffffff' : '#2b0a0a',
                  border: isVisuallySelected ? 'none' : '1px solid rgba(115, 0, 0, 0.2)',
                  boxShadow: isVisuallySelected ? '0 6px 18px rgba(80, 0, 0, 0.25)' : 'none',
                  transition: 'all 0.2s ease'
                }}
                className="transition-all"
              >
                {model.replace('_', ' ').toUpperCase()}
              </button>
            );
          })}
          {/* CMAB button - appears selected when DQN is active */}
          <button
            style={{
              padding: '0.4rem 1.2rem',
              borderRadius: '999px',
              fontSize: '0.85rem',
              fontWeight: '500',
              cursor: 'pointer',
              background: currentModel === 'dqn' ? '#500000' : 'rgba(115, 0, 0, 0.08)',
              color: currentModel === 'dqn' ? '#ffffff' : '#2b0a0a',
              border: currentModel === 'dqn' ? 'none' : '1px solid rgba(115, 0, 0, 0.2)',
              boxShadow: currentModel === 'dqn' ? '0 6px 18px rgba(80, 0, 0, 0.25)' : 'none',
              transition: 'all 0.2s ease'
            }}
            className="transition-all"
          >
            CMAB
          </button>
          {/* SAC button - always unselected appearance */}
          <button
            style={{
              padding: '0.4rem 1.2rem',
              borderRadius: '999px',
              fontSize: '0.85rem',
              fontWeight: '500',
              cursor: 'pointer',
              background: 'rgba(115, 0, 0, 0.08)',
              color: '#2b0a0a',
              border: '1px solid rgba(115, 0, 0, 0.2)',
              boxShadow: 'none',
              transition: 'all 0.2s ease'
            }}
            className="transition-all"
          >
            SAC
          </button>
        </div>
      </div>
    </motion.div>
  )
}

export default ModelSelector

