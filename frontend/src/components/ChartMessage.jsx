// Renders backend-provided chart data inline using recharts.
// Only mounted when the backend returns a `chart` payload (explicit chart request).
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

// Accessible-ish categorical palette.
const COLORS = ['#aa3bff', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#14b8a6', '#8b5cf6', '#ec4899'];

export default function ChartMessage({ chart }) {
  if (!chart || !Array.isArray(chart.data) || chart.data.length === 0) return null;

  const { type, title, data, x_label, y_label } = chart;

  return (
    <div className="chart-card">
      {title && <div className="chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={280}>
        {type === 'pie' ? (
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="label"
              cx="50%"
              cy="50%"
              outerRadius={95}
              label
            >
              {data.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        ) : type === 'line' ? (
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="label" label={x_label ? { value: x_label, position: 'insideBottom', offset: -4 } : undefined} />
            <YAxis label={y_label ? { value: y_label, angle: -90, position: 'insideLeft' } : undefined} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#aa3bff" strokeWidth={2} dot />
          </LineChart>
        ) : (
          <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="label" label={x_label ? { value: x_label, position: 'insideBottom', offset: -4 } : undefined} />
            <YAxis label={y_label ? { value: y_label, angle: -90, position: 'insideLeft' } : undefined} />
            <Tooltip />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
