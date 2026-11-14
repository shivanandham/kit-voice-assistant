import { TooltipProvider } from "@/components/ui/tooltip"
import { AssistantPage } from "@/pages/assistant"
import { HistoryPage } from "@/pages/history"
import { HomePage } from "@/pages/home"
import { LoginPage } from "@/pages/login"

/** The server serves the same index for /, /login, /assistant and /history. */
export default function App() {
  const path = location.pathname.replace(/\/+$/, "") || "/"
  const page = path === "/login" ? <LoginPage /> : path === "/assistant" ? <AssistantPage />
    : path === "/history" ? <HistoryPage /> : <HomePage />
  return <TooltipProvider>{page}</TooltipProvider>
}
