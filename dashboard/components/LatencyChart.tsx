'use client'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { Packet } from '../hooks/useWebSocket'

export function LatencyChart({ packets }: { packets: Packet[] }) {
  const data = [...packets].reverse().slice(-100).map((p, i) => ({
    i,
    ms: p.latency_ms,
  }))

  return (
    <div className="w-full bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4 shadow-sm">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <XAxis dataKey="i" hide />
          <YAxis unit="ms" width={80} stroke="#a1a1aa" fontSize={12} tickLine={false} axisLine={false} />
          <Tooltip 
            formatter={(v: any) => [`${Number(v).toFixed(3)} ms`, 'Latency']}
            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
            labelStyle={{ display: 'none' }}
          />
          <Line 
            type="monotone" 
            dataKey="ms" 
            dot={false} 
            stroke="#6366f1" 
            strokeWidth={2}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
