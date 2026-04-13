export function AlertBanner({ label, onDismiss }: { label: string | null, onDismiss: () => void }) {
  if (!label) return null
  return (
    <div className="fixed top-0 inset-x-0 z-50 bg-red-600 text-white px-6 py-3 flex justify-between items-center shadow-lg">
      <span className="font-semibold">⚠ Attack detected: {label}</span>
      <button onClick={onDismiss} className="text-white/80 hover:text-white font-medium">Dismiss</button>
    </div>
  )
}
