import { Packet } from '../hooks/useWebSocket'

export function TrafficLog({ packets }: { packets: Packet[] }) {
  return (
    <div className="overflow-auto h-72 border rounded-lg text-sm font-mono shadow-sm bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800">
      <table className="w-full">
        <thead className="sticky top-0 bg-white dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800 shadow-sm">
          <tr>
            <th className="text-left px-4 py-3 font-medium text-zinc-600 dark:text-zinc-300">Time</th>
            <th className="text-left px-4 py-3 font-medium text-zinc-600 dark:text-zinc-300">Label</th>
            <th className="text-right px-4 py-3 font-medium text-zinc-600 dark:text-zinc-300">Latency (ms)</th>
          </tr>
        </thead>
        <tbody>
          {packets.map((p, i) => (
            <tr key={i} className={`border-b last:border-b-0 border-zinc-100 dark:border-zinc-800 transition-colors ${p.is_attack ? 'bg-red-50/50 dark:bg-red-950/20' : 'hover:bg-zinc-50 dark:hover:bg-zinc-800/50'}`}>
              <td className="px-4 py-2 text-zinc-400 dark:text-zinc-500">{new Date(p.ts).toLocaleTimeString(undefined, { hour12: false, fractionalSecondDigits: 3 })}</td>
              <td className={`px-4 py-2 font-medium ${p.is_attack ? 'text-red-500' : 'text-emerald-500'}`}>{p.label}</td>
              <td className="px-4 py-2 text-right text-zinc-500 dark:text-zinc-400">{p.latency_ms.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
