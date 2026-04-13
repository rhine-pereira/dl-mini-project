'use client'
import { useWebSocket } from '../hooks/useWebSocket'
import { AlertBanner }  from '../components/AlertBanner'
import { TrafficLog }   from '../components/TrafficLog'
import { LatencyChart } from '../components/LatencyChart'

export default function Page() {
  const { packets, latestAlert, clearAlert } = useWebSocket('ws://localhost:8000/ws/stream')

  return (
    <main className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-50 font-sans p-6 md:p-12 space-y-8">
      <AlertBanner label={latestAlert} onDismiss={clearAlert} />

      <header className="mb-10">
        <h1 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white">Edge IDS Live Dashboard</h1>
        <p className="text-zinc-500 mt-2 text-sm max-w-2xl">Streaming quantized 1D-CNN + GRU model inferences over WebSocket. Displaying network flow classifications and system inference latency locally.</p>
      </header>

      <div className="grid gap-8 lg:grid-cols-2">
        <section className="space-y-3">
          <h2 className="text-lg font-semibold border-b border-zinc-200 dark:border-zinc-800 pb-2">Inference Latency</h2>
          <LatencyChart packets={packets} />
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold border-b border-zinc-200 dark:border-zinc-800 pb-2">Live Traffic Log</h2>
          <TrafficLog packets={packets} />
        </section>
      </div>
    </main>
  )
}
