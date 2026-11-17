export type Source = {
  numero: number
  titre: string
  pages: string
  url: string
  extraits: string[]
  verifie: boolean
  version: string | null
}
export type Reponse = {
  texte: string
  refus: boolean
  raison: string
  sources: Source[]
  cout_usd: number | null
  fiche: string
}
export type Sante = { etat: 'en_service' | 'degrade' | 'arrete' }
export type Vitrine = { chiffres: { libelle: string; valeur: number }[] }

async function lire<T>(reponse: Response): Promise<T> {
  if (!reponse.ok) throw new Error(`Erreur ${reponse.status}`)
  return reponse.json() as Promise<T>
}

export const sante = () => fetch('/health').then(lire<Sante>)
export const vitrine = () => fetch('/vitrine.json').then(lire<Vitrine>)
export const demander = (question: string) =>
  fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  }).then(lire<Reponse>)
