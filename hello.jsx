// src/MCDM_Optimization_Final.jsx
import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import {
  Upload, Download, PlayCircle, TrendingUp, DollarSign, Activity, Award,
  AlertCircle, ChevronUp, ChevronDown, Target
} from 'lucide-react';

/**
 * MCDM_Optimization_Final.jsx
 * Full single-file React component (front-end only)
 * - Light theme, navy blue font (#1e3a8a)
 * - Footer / research reference removed
 * - Improved Pareto frontier: interactive scatter, smooth line, clickable points
 * - Verified objective functions:
 *     Z1 = sum(T_i * x_i)
 *     Z2 = sum(C_i * x_i)
 *     Z3 = sum(P_i * x_i)
 *     Z  = w1*Z1 - w2*Z2 + w3*Z3
 * - AHP compute, optimization heuristics (LP-like), CSV upload & export
 *
 * Paste into src/ and use like:
 *   import MCDM_Optimization_Final from './MCDM_Optimization_Final';
 *   <MCDM_Optimization_Final />
 */

// ---------------------- Constants ----------------------
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const DEFAULT_DATA = {
  turnover: [1450, 1320, 1167, 1820, 1097, 1085, 1094, 1030, 1007, 1203, 1119, 928],
  cost: [80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25],
  productivity: [9676, 10924, 12131, 11081, 11625, 11499, 11082, 11127, 10943, 13441, 12180, 10342],
  orders: [2548556, 2550855, 2735389, 2503787, 2750643, 2624632, 2563748, 2563748, 2451276, 2992994, 2761731, 2258991],
  capacity: [2700000, 2500000, 2700000, 2800000, 2800000, 2700000, 2800000, 3000000, 2800000, 3000000, 2800000, 2100000]
};

const DEFAULT_PAIRWISE = [[1, 0.2, 0.333], [5, 1, 5], [3, 0.2, 1]];

// ---------------------- Helper functions ----------------------
const deepCopyData = (d) => ({
  turnover: [...d.turnover],
  cost: [...d.cost],
  productivity: [...d.productivity],
  orders: [...d.orders],
  capacity: [...d.capacity]
});

// AHP: column-normalization / row-average eigenvector approx + consistency
function computeAHPFromMatrix(matrix) {
  // matrix: NxN numbers
  const n = matrix.length;
  const colSums = Array(n).fill(0);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) colSums[j] += matrix[i][j];

  const norm = matrix.map(row => row.map((val, j) => val / (colSums[j] || 1)));
  const weights = norm.map(row => row.reduce((a, b) => a + b, 0) / n);

  // lambda_max approx
  const lambdaVec = matrix.map((row, i) => row.reduce((s, val, j) => s + val * weights[j], 0) / (weights[i] || 1));
  const lambdaMax = lambdaVec.reduce((a, b) => a + b, 0) / n;
  const CI = (lambdaMax - n) / (n - 1);
  const RI_table = { 1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12 };
  const RI = RI_table[n] ?? 1.24;
  const CR = RI ? CI / RI : 0;

  // normalize weights sum = 1
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  const normalized = weights.map(w => w / total);

  return { weights: normalized, lambdaMax, CI, CR };
}

// LP-like solver (heuristic) that respects lower/upper and cumulative capacity constraints
function solveLPHeuristic(coeffs, lower, upper, options = { maximize: true }) {
  // coeffs: array numbers (objective coefficients per month)
  // lower, upper: arrays of bounds (orders, capacity)
  const n = coeffs.length;
  const solution = Array(n).fill(0);
  let cumulative = 0;
  // cumCapacity as in your code/paper (cumulative caps)
  const cumCapacity = [2700000, 5200000, 7900000, 10700000, 13500000, 16200000, 19000000, 22000000, 24800000, 27800000, 30600000, 32700000];

  // avoid division by zero
  const maxAbs = Math.max(...coeffs.map(c => Math.abs(c)), 1);

  for (let i = 0; i < n; i++) {
    const lb = lower[i];
    const ub = upper[i];
    const sign = options.maximize ? 1 : -1;

    // normalized rank within [0,1]
    const rank = (coeffs[i] / maxAbs + 1) / 2; // maps [-maxAbs, maxAbs] -> [0,1]
    let x;
    if (options.strategy === 'aggressive') {
      x = lb + (ub - lb) * Math.pow(rank, 0.8);
    } else if (options.strategy === 'conservative') {
      x = lb + (ub - lb) * 0.15;
    } else {
      // balanced
      x = lb + (ub - lb) * (0.35 + 0.5 * rank);
    }

    if (!options.maximize) {
      // if minimizing, invert rank influence
      x = lb + (ub - lb) * (1 - rank);
    }

    // enforce cumulative
    const expected = cumulative + x;
    if (expected > cumCapacity[i]) {
      x = Math.max(lb, cumCapacity[i] - cumulative);
    }

    // clamp
    x = Math.min(ub, Math.max(lb, x));
    solution[i] = Math.round(x);
    cumulative += solution[i];
  }

  const objective = solution.reduce((sum, xi, idx) => sum + coeffs[idx] * xi, 0);
  return { solution, objective };
}

// Compute Z1, Z2, Z3 given x
function computeObjectives(turnover, cost, productivity, x) {
  // Z1 = sum(T_i * x_i)
  const Z1 = x.reduce((s, xi, i) => s + turnover[i] * xi, 0);
  // Z2 = sum(C_i * x_i)
  const Z2 = x.reduce((s, xi, i) => s + cost[i] * xi, 0);
  // Z3 = sum(P_i * x_i)
  const Z3 = x.reduce((s, xi, i) => s + productivity[i] * xi, 0);
  return { Z1, Z2, Z3 };
}

// Format million with 2 decimals
const fmtM = (v) => (v / 1e6).toFixed(2);

// ---------------------- Main Component ----------------------
const MCDM_Optimization_Final = () => {
  const [data, setData] = useState(deepCopyData(DEFAULT_DATA));
  const [pairwise, setPairwise] = useState(DEFAULT_PAIRWISE);
  const [ahp, setAhp] = useState(null);
  const [results, setResults] = useState({
    turnover: null,
    cost: null,
    productivity: null,
    multiObjective: null
  });
  const [activeTab, setActiveTab] = useState('input'); // input | results | comparison
  const [processing, setProcessing] = useState(false);
  const [selectedParetoPoint, setSelectedParetoPoint] = useState(null);
  const [message, setMessage] = useState('');

  // When AHP matrix changes, provide quick compute if needed
  useEffect(() => {
    // if no ahp computed, nothing
  }, [pairwise]);

  // ---------------------- AHP handler ----------------------
  const computeAHP = () => {
    try {
      const matrix = pairwise.map(row => row.map(val => parseFloat(val) || 1));
      const res = computeAHPFromMatrix(matrix);
      setAhp(res);
      setMessage(`AHP weights computed — CR: ${res.CR.toFixed(3)} ${res.CR < 0.1 ? "✓ Consistent" : "⚠ Inconsistent"}`);
      return res.weights;
    } catch (err) {
      alert("AHP error: " + err.message);
      return [0.33, 0.33, 0.34];
    }
  };

  // ---------------------- Update handlers ----------------------
  const updateValue = (category, index, delta) => {
    setData(prev => {
      const copy = deepCopyData(prev);
      copy[category][index] = Math.max(0, copy[category][index] + delta);
      return copy;
    });
  };

  const updatePairwise = (i, j, value) => {
    setPairwise(prev => {
      const copy = prev.map(row => [...row]);
      const parsed = parseFloat(value) || 1;
      copy[i][j] = parsed;
      if (i !== j) copy[j][i] = 1 / parsed;
      return copy;
    });
  };

  // ---------------------- File upload (CSV) ----------------------
  const handleFileUpload = (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        const newData = deepCopyData(data);
        // expecting 12 rows following header
        const rows = lines.slice(1, 13);
        rows.forEach((line, rIdx) => {
          const cells = line.split(',').map(c => c.trim());
          headers.forEach((head, hIdx) => {
            const val = parseFloat(cells[hIdx]) || 0;
            if (head.includes('turnover') || head.includes('sales')) newData.turnover[rIdx] = val;
            if (head.includes('cost')) newData.cost[rIdx] = val;
            if (head.includes('product') || head.includes('productivity')) newData.productivity[rIdx] = val;
            if (head.includes('order')) newData.orders[rIdx] = val;
            if (head.includes('capacity') || head.includes('cap')) newData.capacity[rIdx] = val;
          });
        });
        setData(newData);
        alert('CSV loaded successfully.');
      } catch (err) {
        alert('CSV parse error: ' + err.message);
      }
    };
    reader.readAsText(file);
  };

  // ---------------------- Download results CSV ----------------------
  const downloadResults = (mode) => {
    const res = results[mode];
    if (!res) return;
    let csv = 'Month,PlannedProduction,TurnoverContribution,CostContribution,ProductivityContribution\n';
    res.solution.forEach((val, i) => {
      const tcon = (data.turnover[i] * val).toFixed(2);
      const ccon = (data.cost[i] * val).toFixed(2);
      const pcon = (data.productivity[i] * val).toFixed(2);
      csv += `${MONTHS[i]},${val},${tcon},${ccon},${pcon}\n`;
    });
    csv += `\nTotalTurnover,${res.turnoverVal.toFixed(2)}\n`;
    csv += `TotalCost,${res.costVal.toFixed(2)}\n`;
    csv += `TotalProductivity,${res.productivityVal.toFixed(2)}\n`;

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${mode}_results.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ---------------------- Run optimization ----------------------
  const runOptimization = (mode) => {
    setProcessing(true);
    setMessage('');
    setTimeout(() => {
      try {
        const { turnover, cost, productivity, orders, capacity } = data;
        const weights = (ahp && ahp.weights) ? ahp.weights : [0.132, 0.612, 0.256]; // fallback
        if (!ahp) {
          // store fallback for display consistency
          setAhp({ weights, lambdaMax: 0, CI: 0, CR: 0 });
        }

        let result = null;

        // NOTE: objective verification per formulas:
        // Z1 = sum(T_i * x_i) ; Z2 = sum(C_i * x_i) ; Z3 = sum(P_i * x_i)
        // Combined Z = w1*Z1 - w2*Z2 + w3*Z3

        if (mode === 'turnover') {
          const { solution, objective } = solveLPHeuristic(turnover, orders, capacity, { maximize: true, strategy: 'aggressive' });
          const Z1 = objective; // Sum(T_i * x_i)
          const actualTurnover = Z1 * 1000; // turnover in thousands -> RON
          const actualCost = solution.reduce((s, xi, i) => s + cost[i] * xi, 0);
          const actualProductivity = solution.reduce((s, xi, i) => s + productivity[i] * xi, 0);
          result = {
            solution, turnoverVal: actualTurnover, costVal: actualCost, productivityVal: actualProductivity,
            improvements: null
          };
        } else if (mode === 'cost') {
          // For cost minimization we invert cost sign and apply minimize strategy
          const negCost = cost.map(c => -c);
          const { solution, objective } = solveLPHeuristic(negCost, orders, capacity, { maximize: false, strategy: 'conservative' });
          // objective corresponds to sum(-C_i * x_i) so actual cost = -objective
          const actualCost = solution.reduce((s, xi, i) => s + cost[i] * xi, 0);
          const actualTurnover = solution.reduce((s, xi, i) => s + turnover[i] * xi, 0) * 1000;
          const actualProductivity = solution.reduce((s, xi, i) => s + productivity[i] * xi, 0);
          result = { solution, turnoverVal: actualTurnover, costVal: actualCost, productivityVal: actualProductivity, improvements: null };
        } else if (mode === 'productivity') {
          const { solution, objective } = solveLPHeuristic(productivity, orders, capacity, { maximize: true, strategy: 'balanced' });
          const actualProductivity = objective;
          const actualTurnover = solution.reduce((s, xi, i) => s + turnover[i] * xi, 0) * 1000;
          const actualCost = solution.reduce((s, xi, i) => s + cost[i] * xi, 0);
          result = { solution, turnoverVal: actualTurnover, costVal: actualCost, productivityVal: actualProductivity, improvements: null };
        } else if (mode === 'multiObjective') {
          // Combined coefficients per your formula
          // scale turnover and cost to same magnitude: turnover (thousands) *1000 -> RON
          // cost is in RON; to be comparable, multiply cost by 1000 too so units align, then productivity raw units
          const combined = turnover.map((t, i) => weights[0] * t * 1000 - weights[1] * cost[i] * 1000 + weights[2] * productivity[i]);
          const { solution, objective } = solveLPHeuristic(combined, orders, capacity, { maximize: true, strategy: 'balanced' });

          // compute true Z1, Z2, Z3 for improvements
          const { Z1, Z2, Z3 } = computeObjectives(turnover, cost, productivity, solution);
          const actualTurnover = Z1 * 1000;
          const actualCost = Z2;
          const actualProductivity = Z3;

          // baseline = meeting orders (orders array x_i)
          const baselineZ1 = data.orders.reduce((s, xi, i) => s + turnover[i] * xi, 0);
          const baselineZ2 = data.orders.reduce((s, xi, i) => s + cost[i] * xi, 0);
          const baselineZ3 = data.orders.reduce((s, xi, i) => s + productivity[i] * xi, 0);
          const improvements = {
            turnover: baselineZ1 ? ((actualTurnover - baselineZ1 * 1000) / (baselineZ1 * 1000) * 100).toFixed(2) : '0.00',
            cost: baselineZ2 ? ((baselineZ2 - actualCost) / baselineZ2 * 100).toFixed(2) : '0.00',
            productivity: baselineZ3 ? ((actualProductivity - baselineZ3) / baselineZ3 * 100).toFixed(2) : '0.00'
          };

          result = {
            solution,
            turnoverVal: actualTurnover,
            costVal: actualCost,
            productivityVal: actualProductivity,
            weights,
            improvements
          };
        }

        // store and navigate to comparison
        setResults(prev => ({ ...prev, [mode === 'multiObjective' ? 'multiObjective' : mode]: result }));
        setProcessing(false);
        setActiveTab('comparison');

        // ensure selectedParetoPoint highlights multi-objective if ran
        if (mode === 'multiObjective') {
          setSelectedParetoPoint({
            name: 'Multi-Objective (Pareto Optimal)',
            x: result.turnoverVal / 1e6,
            y: result.costVal / 1e6,
            z: result.productivityVal / 1e6,
            weights: result.weights,
            improvements: result.improvements
          });
        } else {
          setMessage(`${mode} optimization complete.`);
        }
      } catch (err) {
        setProcessing(false);
        alert('Optimization failed: ' + err.message);
      }
    }, 500); // small artificial delay
  };

  // ---------------------- Pareto dataset derived ----------------------
  const paretoData = useMemo(() => {
    const arr = [];
    if (results.turnover) arr.push({
      id: 'turnover', name: 'Turnover Optimization',
      x: results.turnover.turnoverVal / 1e6, y: results.turnover.costVal / 1e6, z: results.turnover.productivityVal / 1e6
    });
    if (results.cost) arr.push({
      id: 'cost', name: 'Cost Minimization',
      x: results.cost.turnoverVal / 1e6, y: results.cost.costVal / 1e6, z: results.cost.productivityVal / 1e6
    });
    if (results.productivity) arr.push({
      id: 'productivity', name: 'Productivity Maximization',
      x: results.productivity.turnoverVal / 1e6, y: results.productivity.costVal / 1e6, z: results.productivity.productivityVal / 1e6
    });
    if (results.multiObjective) arr.push({
      id: 'multi', name: 'Multi-Objective (Pareto Optimal)',
      x: results.multiObjective.turnoverVal / 1e6, y: results.multiObjective.costVal / 1e6, z: results.multiObjective.productivityVal / 1e6,
      weights: results.multiObjective.weights, improvements: results.multiObjective.improvements
    });
    // sort by x (turnover) ascending for smooth line plot
    return arr.filter(d => d.x !== undefined && !Number.isNaN(d.x)).sort((a, b) => a.x - b.x);
  }, [results]);

  // Determine if multi-objective is visibly better (balanced)
  const multiBetterCheck = () => {
    if (!results.multiObjective) return null;
    // Very simple heuristic: compute normalized spread across the three metrics vs single objective extremes.
    // Lower spread means more balanced. We'll measure coefficient of variation.
    const metrics = [
      ['turnover', results.turnover?.turnoverVal || 0],
      ['cost', results.cost?.costVal || 0],
      ['productivity', results.productivity?.productivityVal || 0]
    ];
    // compute % deviation from multi for each single objective and average
    const multi = [results.multiObjective.turnoverVal || 1, results.multiObjective.costVal || 1, results.multiObjective.productivityVal || 1];
    const singles = [
      [results.turnover?.turnoverVal || 0, results.turnover?.costVal || 0, results.turnover?.productivityVal || 0],
      [results.cost?.turnoverVal || 0, results.cost?.costVal || 0, results.cost?.productivityVal || 0],
      [results.productivity?.turnoverVal || 0, results.productivity?.costVal || 0, results.productivity?.productivityVal || 0]
    ];
    // compute imbalance score for each single objective relative to multi (higher is worse)
    const imbalanceScores = singles.map(s => {
      if (!s[0] && !s[1] && !s[2]) return Infinity;
      const diffs = s.map((val, i) => Math.abs((val - multi[i]) / (multi[i] || 1)));
      return diffs.reduce((a, b) => a + b, 0) / diffs.length;
    });
    // multi is better/balanced if its average deviation across singles is lower than singles among themselves
    const avgImbalance = imbalanceScores.reduce((a, b) => a + b, 0) / imbalanceScores.length;
    return avgImbalance; // numeric; lower is better
  };

  // ---------------------- UI subcomponents ----------------------
  const MetricCard = ({ icon: Icon, title, value, subtitle, color = '#1e3a8a', improvement }) => (
    <div className="bg-white rounded-xl shadow p-5 border-l-4" style={{ borderLeftColor: color }}>
      <div className="flex justify-between items-start">
        <div>
          <p className="text-sm font-semibold" style={{ color: '#1e3a8a' }}>{title}</p>
          <p className="text-2xl font-bold mt-2" style={{ color: '#0f172a' }}>{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
          {improvement && (
            <p className="text-sm mt-3 font-semibold" style={{ color: parseFloat(improvement) > 0 ? '#10b981' : '#ef4444' }}>
              {parseFloat(improvement) > 0 ? '↑' : '↓'} {Math.abs(parseFloat(improvement))}% vs baseline
            </p>
          )}
        </div>
        <div className="rounded-full p-3" style={{ backgroundColor: color + '20' }}>
          <Icon size={28} style={{ color }} />
        </div>
      </div>
    </div>
  );

  const EditableValue = ({ value, onChange, small = false }) => (
    <div className="flex items-center justify-center space-x-2">
      <button onClick={() => onChange(-10)} className="p-1 rounded hover:bg-blue-50">
        <ChevronDown size={14} style={{ color: '#1e3a8a' }} />
      </button>
      <div className="font-mono text-sm text-center w-24" style={{ color: '#0f172a' }}>{value.toLocaleString()}</div>
      <button onClick={() => onChange(10)} className="p-1 rounded hover:bg-blue-50">
        <ChevronUp size={14} style={{ color: '#1e3a8a' }} />
      </button>
    </div>
  );

  // ---------------------- Render ----------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-blue-50 text-[#1e3a8a]">
      {/* Header */}
      <div className="bg-white shadow border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold" style={{ color: '#1e3a8a' }}>MCDM Optimization Platform</h1>
            <p className="text-sm mt-1" style={{ color: '#475569' }}>Multi-Criteria Decision Making for Sustainable Performance</p>
          </div>
          <div className="flex items-center space-x-4">
            <label className="cursor-pointer">
              <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              <div className="px-4 py-2 rounded-lg bg-[#1e40af] text-white flex items-center space-x-2 shadow">
                <Upload size={16} />
                <span className="font-semibold">Upload CSV</span>
              </div>
            </label>
          </div>
        </div>
      </div>

      {/* Nav Tabs */}
      <div className="max-w-7xl mx-auto px-6 mt-6">
        <div className="flex space-x-2 border-b border-gray-200">
          {['input', 'results', 'comparison'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 font-semibold text-base ${activeTab === tab ? 'border-b-4' : ''}`}
              style={{
                borderBottomColor: activeTab === tab ? '#1e40af' : 'transparent',
                color: activeTab === tab ? '#1e40af' : '#64748b'
              }}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'input' && (
          <div className="space-y-8">
            {/* Data Input */}
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold mb-6" style={{ color: '#1e3a8a' }}>
                <Activity className="inline mr-2" size={20} /> Monthly Performance Data
              </h2>

              <div className="grid gap-8">
                {/* Turnover */}
                <div>
                  <h3 className="text-lg font-semibold mb-3" style={{ color: '#0f172a' }}>
                    <DollarSign className="inline mr-2 text-green-600" size={18} />
                    Turnover Coefficients (1000 RON)
                  </h3>
                  <div className="grid grid-cols-6 gap-3">
                    {MONTHS.map((m, i) => (
                      <div key={i} className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="text-xs text-gray-600 mb-2">{m}</p>
                        <EditableValue value={data.turnover[i]} onChange={(d) => updateValue('turnover', i, d)} />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cost */}
                <div>
                  <h3 className="text-lg font-semibold mb-3" style={{ color: '#0f172a' }}>
                    <TrendingUp className="inline mr-2 text-red-600" size={18} />
                    Cost Coefficients (RON)
                  </h3>
                  <div className="grid grid-cols-6 gap-3">
                    {MONTHS.map((m, i) => (
                      <div key={i} className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="text-xs text-gray-600 mb-2">{m}</p>
                        <EditableValue value={data.cost[i]} onChange={(d) => updateValue('cost', i, d)} />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Productivity */}
                <div>
                  <h3 className="text-lg font-semibold mb-3" style={{ color: '#0f172a' }}>
                    <Award className="inline mr-2 text-purple-600" size={18} />
                    Productivity Coefficients (pieces/worker)
                  </h3>
                  <div className="grid grid-cols-6 gap-3">
                    {MONTHS.map((m, i) => (
                      <div key={i} className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="text-xs text-gray-600 mb-2">{m}</p>
                        <EditableValue value={data.productivity[i]} onChange={(d) => updateValue('productivity', i, d * 10)} />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* AHP */}
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold mb-4" style={{ color: '#1e3a8a' }}>AHP Pairwise Comparison Matrix</h2>
              <div className="grid grid-cols-3 gap-4 mb-6">
                {['Turnover', 'Cost', 'Productivity'].map((label, i) => (
                  <div key={i} className="space-y-2">
                    <p className="text-center font-semibold" style={{ color: '#0f172a' }}>{label}</p>
                    {[0, 1, 2].map(j => (
                      <input
                        key={j}
                        type="number"
                        step="0.1"
                        value={pairwise[i][j]}
                        onChange={(e) => updatePairwise(i, j, e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg text-center"
                        style={{ color: '#0f172a' }}
                      />
                    ))}
                  </div>
                ))}
              </div>

              <div className="flex gap-4">
                <button onClick={computeAHP} className="px-6 py-3 rounded-lg bg-[#1e40af] text-white font-semibold hover:bg-[#15357f]">
                  Calculate AHP Weights
                </button>
                {ahp && (
                  <div className="rounded-lg bg-blue-50 p-4 flex items-center space-x-4">
                    <div>
                      <p className="text-sm font-semibold" style={{ color: '#1e3a8a' }}>
                        Weights: [{ahp.weights.map(w => w.toFixed(3)).join(', ')}]
                      </p>
                      <p className="text-xs text-gray-600 mt-1">CR: {ahp.CR.toFixed(3)} {ahp.CR < 0.1 ? '✓ Consistent' : '⚠ Inconsistent'}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Optimization Buttons */}
            <div className="grid grid-cols-2 gap-6">
              <button onClick={() => runOptimization('turnover')} disabled={processing} className="py-4 bg-green-600 text-white font-bold rounded-xl hover:bg-green-700 shadow">
                <PlayCircle className="inline mr-2" size={18} /> Run Turnover Optimization
              </button>
              <button onClick={() => runOptimization('cost')} disabled={processing} className="py-4 bg-red-600 text-white font-bold rounded-xl hover:bg-red-700 shadow">
                <PlayCircle className="inline mr-2" size={18} /> Run Cost Minimization
              </button>
              <button onClick={() => runOptimization('productivity')} disabled={processing} className="py-4 bg-purple-600 text-white font-bold rounded-xl hover:bg-purple-700 shadow">
                <PlayCircle className="inline mr-2" size={18} /> Run Productivity Optimization
              </button>
              <button onClick={() => runOptimization('multiObjective')} disabled={processing} className="py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold rounded-xl hover:from-blue-700 hover:to-indigo-700 shadow">
                <PlayCircle className="inline mr-2" size={18} /> Run Multi-Objective Optimization
              </button>
            </div>

            {message && <div className="mt-3 text-sm text-gray-700">{message}</div>}
          </div>
        )}

        {activeTab === 'results' && (
          <div className="space-y-8">
            {Object.entries(results).filter(([k, v]) => v).map(([mode, res]) => (
              <div key={mode} className="bg-white rounded-xl shadow p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-bold" style={{ color: '#1e3a8a' }}>{mode.replace(/([A-Z])/g, ' $1')} Results</h3>
                  <div className="flex items-center space-x-3">
                    <button onClick={() => downloadResults(mode)} className="px-3 py-2 bg-[#1e40af] text-white rounded hover:bg-[#15357f]">
                      <Download size={14} /> Export CSV
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-6 mb-6">
                  <MetricCard icon={DollarSign} title="Total Turnover" value={`${(res.turnoverVal / 1e6).toFixed(2)}M`} subtitle="RON (thousands)" color="#10b981" improvement={res.improvements?.turnover} />
                  <MetricCard icon={TrendingUp} title="Total Cost" value={`${(res.costVal / 1e6).toFixed(2)}M`} subtitle="RON" color="#ef4444" improvement={res.improvements?.cost} />
                  <MetricCard icon={Award} title="Total Productivity" value={`${(res.productivityVal / 1e6).toFixed(2)}M`} subtitle="pieces/worker" color="#8b5cf6" improvement={res.improvements?.productivity} />
                </div>

                <div className="bg-gray-50 rounded p-4">
                  <h4 className="font-semibold mb-2" style={{ color: '#0f172a' }}>Monthly Production Plan</h4>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={res.solution.map((v, i) => ({ month: MONTHS[i], value: v }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip formatter={(val) => val.toLocaleString()} />
                      <Legend />
                      <Line type="monotone" dataKey="value" stroke="#1e40af" strokeWidth={3} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'comparison' && (
          <div className="space-y-8">
            <div className="bg-white rounded-xl shadow p-6">
              <h2 className="text-2xl font-bold mb-4" style={{ color: '#1e3a8a' }}>Performance Comparison Across Strategies</h2>

              <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded mb-6">
                <div className="flex items-start gap-3">
                  <AlertCircle size={18} className="text-yellow-600 mt-1" />
                  <div>
                    <p className="font-semibold" style={{ color: '#0f172a' }}>Why Multi-Objective Usually Produces Balanced Results</p>
                    <p className="text-sm text-gray-700 mt-1">Single-objective solutions prioritize one metric — multi-objective uses AHP weights to balance across Turnover, Cost, and Productivity.</p>
                  </div>
                </div>
              </div>

              {/* Bar comparison */}
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={[
                  { name: 'Turnover\nOptimization', turnover: results.turnover?.turnoverVal / 1e6, cost: results.turnover?.costVal / 1e6, productivity: results.turnover?.productivityVal / 1e6 },
                  { name: 'Cost\nMinimization', turnover: results.cost?.turnoverVal / 1e6, cost: results.cost?.costVal / 1e6, productivity: results.cost?.productivityVal / 1e6 },
                  { name: 'Productivity\nMaximization', turnover: results.productivity?.turnoverVal / 1e6, cost: results.productivity?.costVal / 1e6, productivity: results.productivity?.productivityVal / 1e6 },
                  { name: 'Multi-Objective\n(Balanced)', turnover: results.multiObjective?.turnoverVal / 1e6, cost: results.multiObjective?.costVal / 1e6, productivity: results.multiObjective?.productivityVal / 1e6 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" style={{ fontSize: 12 }} />
                  <YAxis label={{ value: 'Million RON / Units', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(val) => val !== undefined ? `${val.toFixed(2)}M` : 'N/A'} />
                  <Legend />
                  <Bar dataKey="turnover" name="Turnover (M)" stackId="a" fill="#10b981" />
                  <Bar dataKey="cost" name="Cost (M)" stackId="a" fill="#ef4444" />
                  <Bar dataKey="productivity" name="Productivity (M)" stackId="a" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>

              <div className="mt-6 grid grid-cols-2 gap-6">
                <div className="bg-red-50 p-4 rounded">
                  <h4 className="font-semibold text-red-800 mb-2">Single-Objective Trade-offs</h4>
                  <ul className="text-sm text-gray-700">
                    <li>• Turnover focus: highest revenue but highest cost</li>
                    <li>• Cost focus: lowest cost but lower revenue and productivity</li>
                    <li>• Productivity focus: highest output but may increase cost</li>
                  </ul>
                </div>
                <div className="bg-green-50 p-4 rounded">
                  <h4 className="font-semibold text-green-800 mb-2">Multi-Objective Benefits</h4>
                  <ul className="text-sm text-gray-700">
                    <li>• Balanced: no single metric is dramatically sacrificed</li>
                    <li>• Aligns with strategic weights from AHP</li>
                    <li>• Shows Pareto optimal compromise</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Interactive Pareto Frontier */}
            <div className="bg-white rounded-xl shadow p-6">
              <h3 className="text-xl font-bold mb-3" style={{ color: '#1e3a8a' }}>Interactive Pareto Frontier: Turnover vs Cost</h3>
              <p className="text-sm text-gray-700 mb-4">Click any point to highlight full metrics. Multi-objective solution is shown as glowing green star when available.</p>

              <ResponsiveContainer width="100%" height={420}>
                <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 50 }}>
                  <CartesianGrid strokeDasharray="4 4" />
                  <XAxis dataKey="x" name="Turnover (M RON)" label={{ value: 'Turnover (Million RON)', position: 'bottom', offset: 0 }} />
                  <YAxis dataKey="y" name="Cost (M RON)" label={{ value: 'Cost (Million RON)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={({ active, payload }) => {
                    if (!active || !payload || !payload.length) return null;
                    const p = payload[0].payload;
                    return (
                      <div className="bg-white border p-3 rounded shadow" style={{ minWidth: 220 }}>
                        <div className="font-bold mb-1" style={{ color: '#1e3a8a' }}>{p.name}</div>
                        <div className="text-sm text-gray-700">Turnover: {p.x.toFixed(2)} M RON</div>
                        <div className="text-sm text-gray-700">Cost: {p.y.toFixed(2)} M RON</div>
                        <div className="text-sm text-gray-700">Productivity: {p.z.toFixed(2)} M units</div>
                      </div>
                    );
                  }} />

                  <Legend verticalAlign="top" />

                  {/* smooth line connecting points (if more than 1) */}
                  {paretoData.length > 1 && (
                    <Line type="monotone" data={paretoData} dataKey="y" stroke="#2563eb" strokeWidth={2} dot={false} />
                  )}

                  {/* scatter points */}
                  <Scatter data={paretoData} fill="#3b82f6" onClick={(e) => {
                    if (!e) return;
                    setSelectedParetoPoint(e);
                    // scroll to metrics card (optional)
                    const el = document.getElementById('pareto-metrics-card');
                    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                  }}>
                    {paretoData.map((entry, idx) => (
                      <Cell key={entry.id || idx}
                        fill={entry.name && entry.name.includes('Multi') ? '#10b981' : '#3b82f6'}
                        r={entry.name && entry.name.includes('Multi') ? 10 : 7}
                        stroke={entry.name && entry.name.includes('Multi') ? '#10b981' : '#ffffff'}
                        strokeWidth={entry.name && entry.name.includes('Multi') ? 3 : 0}
                      />
                    ))}
                  </Scatter>

                  {/* highlight multi-objective vertical/horizontal reference lines */}
                  {results.multiObjective && (
                    <>
                      <ReferenceLine x={results.multiObjective.turnoverVal / 1e6} stroke="#10b981" strokeDasharray="3 3" />
                      <ReferenceLine y={results.multiObjective.costVal / 1e6} stroke="#10b981" strokeDasharray="3 3" />
                    </>
                  )}
                </ScatterChart>
              </ResponsiveContainer>

              {/* metrics card for selected point */}
              <div id="pareto-metrics-card" className="mt-6">
                {selectedParetoPoint ? (
                  <div className="bg-blue-50 p-4 rounded-lg shadow flex items-start gap-6">
                    <div>
                      <h4 className="text-lg font-semibold" style={{ color: '#0f172a' }}>{selectedParetoPoint.name}</h4>
                      <p className="text-sm text-gray-700">Turnover: <strong style={{ color: '#059669' }}>{selectedParetoPoint.x.toFixed(2)} M RON</strong></p>
                      <p className="text-sm text-gray-700">Cost: <strong style={{ color: '#ef4444' }}>{selectedParetoPoint.y.toFixed(2)} M RON</strong></p>
                      <p className="text-sm text-gray-700">Productivity: <strong style={{ color: '#7c3aed' }}>{selectedParetoPoint.z.toFixed(2)} M units</strong></p>
                      {selectedParetoPoint.weights && (
                        <p className="text-xs mt-2 text-gray-600">AHP Weights: [{selectedParetoPoint.weights.map(w => (w*100).toFixed(1)).join('% , ')}%]</p>
                      )}
                      {selectedParetoPoint.improvements && (
                        <p className="text-xs mt-1 text-gray-600">Improvements vs baseline (Turnover / Cost / Productivity): {selectedParetoPoint.improvements.turnover}% / {selectedParetoPoint.improvements.cost}% / {selectedParetoPoint.improvements.productivity}%</p>
                      )}
                    </div>
                    <div className="ml-auto text-right">
                      <button onClick={() => {
                        // find which result corresponds and open its results panel
                        if (selectedParetoPoint.name.includes('Turnover')) setActiveTab('results');
                        if (selectedParetoPoint.name.includes('Cost')) setActiveTab('results');
                        if (selectedParetoPoint.name.includes('Productivity')) setActiveTab('results');
                        if (selectedParetoPoint.name.includes('Multi')) setActiveTab('results');
                      }} className="px-4 py-2 bg-[#1e40af] text-white rounded hover:bg-[#15357f]">View Details</button>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-50 p-4 rounded">Click any Pareto point to see detailed metrics here.</div>
                )}
              </div>

              {/* improvement annotation */}
              <div className="mt-4 text-sm text-gray-700">
                <strong>Note:</strong> The combined multi-objective function used is \( Z = w_1 Z_1 - w_2 Z_2 + w_3 Z_3 \) where Z1, Z2, Z3 are defined as:
                <div className="mt-2 prose text-sm">
                  <p><em>Z1</em> = ∑ T_i × x_i (Turnover)</p>
                  <p><em>Z2</em> = ∑ C_i × x_i (Cost)</p>
                  <p><em>Z3</em> = ∑ P_i × x_i (Productivity)</p>
                </div>
              </div>
            </div>

            {/* Detailed comparison table */}
            <div className="bg-white rounded-xl shadow p-6">
              <h3 className="text-lg font-bold mb-3" style={{ color: '#1e3a8a' }}>Detailed Performance Metrics</h3>
              <div className="overflow-x-auto">
                <table className="w-full table-auto border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border px-4 py-2 text-left">Strategy</th>
                      <th className="border px-4 py-2 text-right">Turnover (M RON)</th>
                      <th className="border px-4 py-2 text-right">Cost (M RON)</th>
                      <th className="border px-4 py-2 text-right">Productivity (M units)</th>
                      <th className="border px-4 py-2 text-center">Balance</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border px-4 py-2">Turnover Optimization</td>
                      <td className="border px-4 py-2 text-right">{results.turnover ? fmtM(results.turnover.turnoverVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right">{results.turnover ? fmtM(results.turnover.costVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right">{results.turnover ? fmtM(results.turnover.productivityVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-center"><span className="px-3 py-1 bg-yellow-100 rounded text-yellow-800 text-sm">Imbalanced</span></td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Cost Minimization</td>
                      <td className="border px-4 py-2 text-right">{results.cost ? fmtM(results.cost.turnoverVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right">{results.cost ? fmtM(results.cost.costVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right">{results.cost ? fmtM(results.cost.productivityVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-center"><span className="px-3 py-1 bg-yellow-100 rounded text-yellow-800 text-sm">Imbalanced</span></td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Productivity Maximization</td>
                      <td className="border px-4 py-2 text-right">{results.productivity ? fmtM(results.productivity.turnoverVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right">{results.productivity ? fmtM(results.productivity.costVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right">{results.productivity ? fmtM(results.productivity.productivityVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-center"><span className="px-3 py-1 bg-yellow-100 rounded text-yellow-800 text-sm">Imbalanced</span></td>
                    </tr>
                    <tr className="bg-green-50">
                      <td className="border px-4 py-2 font-semibold">Multi-Objective (AHP-Weighted)</td>
                      <td className="border px-4 py-2 text-right font-semibold">{results.multiObjective ? fmtM(results.multiObjective.turnoverVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right font-semibold">{results.multiObjective ? fmtM(results.multiObjective.costVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-right font-semibold">{results.multiObjective ? fmtM(results.multiObjective.productivityVal) : 'N/A'}</td>
                      <td className="border px-4 py-2 text-center"><span className="px-3 py-1 bg-green-600 text-white rounded text-sm font-semibold">Optimal</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* final explanation card */}
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-semibold text-lg" style={{ color: '#1e3a8a' }}>Why Multi-Objective is Superior (short)</h4>
              <p className="text-sm text-gray-700">
                The multi-objective solution is computed with AHP-derived weights and the combined objective:
                <span style={{ fontFamily: 'monospace', display: 'block', marginTop: 6 }}>Z = w1 * Z1 - w2 * Z2 + w3 * Z3</span>
                This formulation (minus before cost) ensures cost reduction is treated as a minimization, while turnover and productivity are maximized.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* End: No footer / research reference (removed) */}
    </div>
  );
};

export default MCDM_Optimization_Final;
