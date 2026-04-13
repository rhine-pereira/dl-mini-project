import { useEffect, useRef, useState } from 'react'

export interface Packet {
  label:      string
  is_attack:  boolean
  latency_ms: number
  ts:         number          // client-side timestamp
}

export function useWebSocket(url: string) {
  const [packets, setPackets] = useState<Packet[]>([])
  const [latestAlert, setLatestAlert] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    ws.current = new WebSocket(url)
    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data) as Omit<Packet, 'ts'>
      const packet: Packet = { ...data, ts: Date.now() }

      setPackets(prev => [packet, ...prev].slice(0, 500))  // keep last 500
      if (data.is_attack) setLatestAlert(data.label)
    }
    
    ws.current.onclose = () => {
        console.log('WebSocket disconnected')
    }

    return () => {
        if(ws.current) {
            ws.current.close()
        }
    }
  }, [url])

  return { packets, latestAlert, clearAlert: () => setLatestAlert(null) }
}
