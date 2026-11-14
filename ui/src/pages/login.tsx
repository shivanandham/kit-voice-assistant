import { useEffect, useState, type FormEvent } from "react"
import { Loader2Icon } from "lucide-react"

import { SiteHeader } from "@/components/kit/site-header"
import { isLoggedIn, loadMe } from "@/lib/me"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { login } from "@/lib/api"

export function LoginPage() {
  const [id, setId] = useState("")
  const [pw, setPw] = useState("")
  const [err, setErr] = useState("")
  const [busy, setBusy] = useState(false)
  useEffect(() => { loadMe().then((m) => { if (isLoggedIn(m)) location.replace("/assistant") }) }, [])
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setErr("")
    setBusy(true)
    try {
      await login(id.trim(), pw)
      location.href = "/assistant"
    } catch (ex) {
      setErr((ex as Error).message || "Login failed.")
      setBusy(false)
    }
  }
  return (
    <>
      <SiteHeader />
      <main className="mx-auto flex max-w-6xl justify-center px-4 py-12 sm:py-20">
        <Card className="w-full max-w-sm">
          <CardHeader>
            <CardTitle><h1 className="text-2xl font-semibold">Log in</h1></CardTitle>
            <CardDescription>Students and teachers sign in with their college login.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={submit} className="flex flex-col gap-4" noValidate={false}>
              <div className="flex flex-col gap-2">
                <Label htmlFor="login_id">Login ID</Label>
                <Input id="login_id" name="login_id" autoComplete="username" required value={id} onChange={(e) => setId(e.target.value)} />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="password">Password</Label>
                <Input id="password" name="password" type="password" autoComplete="current-password" required value={pw} onChange={(e) => setPw(e.target.value)} />
              </div>
              {err && <Alert variant="destructive" role="alert"><AlertDescription>{err}</AlertDescription></Alert>}
              <Button type="submit" disabled={busy} className="h-10">
                {busy && <Loader2Icon aria-hidden className="animate-spin" />} Log in
              </Button>
            </form>
          </CardContent>
        </Card>
      </main>
    </>
  )
}
