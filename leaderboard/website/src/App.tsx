import { useState, useEffect } from 'react';
import { Flame, Activity, Zap, ShieldAlert } from 'lucide-react';
import './index.css';

interface Submission {
  id: string;
  name: string;
  repo_url: string;
  track: string;
  quality_score: number;
  compression_ratio: number;
  latency_ms: number;
  final_score: number;
  roast: string;
  is_reference: boolean;
}

function App() {
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [selectedSubmission, setSelectedSubmission] = useState<Submission | null>(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + 'submissions.json')
      .then(res => res.json())
      .then(data => {
        // Sort by final score descending
        const sorted = data.sort((a: Submission, b: Submission) => b.final_score - a.final_score);
        setSubmissions(sorted);
        if (sorted.length > 0) {
          setSelectedSubmission(sorted[0]);
        }
      });
  }, []);

  const getRankClass = (index: number, isRef: boolean) => {
    if (isRef) return '';
    // Count only non-reference submissions for actual ranking
    const rank = submissions.filter((s, i) => i <= index && !s.is_reference).length;
    if (rank === 1) return 'rank-1';
    if (rank === 2) return 'rank-2';
    if (rank === 3) return 'rank-3';
    return '';
  };

  const getRankDisplay = (index: number, isRef: boolean) => {
    if (isRef) return '-';
    return submissions.filter((s, i) => i <= index && !s.is_reference).length;
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>ContextBench Leaderboard</h1>
        <p>The definitive benchmark for LLM context optimization architectures.</p>
      </header>

      <div className="glass-panel" style={{ marginBottom: '2rem' }}>
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th style={{ width: '80px', textAlign: 'center' }}>Rank</th>
              <th>Submission</th>
              <th>Quality Score</th>
              <th>Compression</th>
              <th>Latency</th>
              <th>Final Score</th>
            </tr>
          </thead>
          <tbody>
            {submissions.map((sub, index) => (
              <tr 
                key={sub.id}
                onClick={() => setSelectedSubmission(sub)}
                style={{ cursor: 'pointer', background: selectedSubmission?.id === sub.id ? 'rgba(59, 130, 246, 0.1)' : '' }}
              >
                <td style={{ textAlign: 'center' }}>
                  <div className={`rank-badge ${getRankClass(index, sub.is_reference)}`}>
                    {getRankDisplay(index, sub.is_reference)}
                  </div>
                </td>
                <td>
                  <div className="submission-name">
                    {sub.name}
                    {sub.is_reference && <span className="reference-badge">Reference</span>}
                  </div>
                  <a 
                    href={sub.repo_url} 
                    className="github-link" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    GitHub Repo
                  </a>
                </td>
                <td className="score-cell">
                  <span style={{ color: sub.quality_score >= 78 ? 'var(--success)' : 'var(--danger)' }}>
                    {sub.quality_score.toFixed(1)}
                  </span>
                </td>
                <td className="score-cell">{(sub.compression_ratio * 100).toFixed(1)}%</td>
                <td className="score-cell">{sub.latency_ms}ms</td>
                <td className="score-cell final-score">{sub.final_score.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedSubmission && (
        <div className="glass-panel" style={{ display: 'flex', gap: '2rem' }}>
          <div style={{ flex: 1 }}>
            <h2 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={24} color="var(--accent)" />
              {selectedSubmission.name} Performance
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.5rem', marginTop: '1.5rem' }}>
              <div>
                <span className="metric-label">QUALITY NORM</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>
                  {(selectedSubmission.quality_score / 100).toFixed(3)}
                </div>
              </div>
              <div>
                <span className="metric-label">COMPRESSION REWARD</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>
                  {(1 - selectedSubmission.compression_ratio).toFixed(3)}
                </div>
              </div>
              <div>
                <span className="metric-label">LATENCY MULTIPLIER</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>
                  {selectedSubmission.latency_ms < 100 ? '1.00' : 
                   selectedSubmission.latency_ms < 500 ? '0.95' : 
                   selectedSubmission.latency_ms < 2000 ? '0.85' : '0.70'}
                </div>
              </div>
            </div>
            
            <div className="roast-box">
              <div className="roast-text">{selectedSubmission.roast}</div>
              <div className="roast-author" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <Flame size={14} color="var(--danger)" /> ContextOps Roast Engine
              </div>
            </div>
          </div>
          
          <div style={{ width: '300px', background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '0.75rem' }}>
            <h3 style={{ marginTop: 0, fontSize: '1rem', color: 'var(--text-secondary)' }}>Evaluation Specs</h3>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              <li style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem' }}>
                <ShieldAlert size={16} color="var(--accent)" />
                <span><strong>Quality Gate:</strong> 78 points (empirically calibrated)</span>
              </li>
              <li style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem' }}>
                <Zap size={16} color="var(--accent)" />
                <span><strong>Formula (v1):</strong> 50% Quality + 35% Compression + 15% Latency</span>
              </li>
              <li style={{ display: 'flex', gap: '0.5rem' }}>
                <Activity size={16} color="var(--accent)" />
                <span><strong>Dataset:</strong> ContextBench_v1 (1,500 samples)</span>
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
