import { BadgeLigne, Bouton, Etat, Logo, Valide } from '@quen9669/lignes/ui'
import { useEffect, useState, type FormEvent } from 'react'
import { demander, sante, vitrine, type Reponse, type Sante, type Vitrine } from './api'

const AVERTISSEMENT = "Cet assistant aide à l'analyse et ne remplace pas un avis juridique."

export function App() {
  const [etat, setEtat] = useState<Sante['etat']>('arrete')
  const [chiffres, setChiffres] = useState<Vitrine['chiffres']>([])
  const [question, setQuestion] = useState('')
  const [reponse, setReponse] = useState<Reponse | null>(null)
  const [erreur, setErreur] = useState('')
  const [attente, setAttente] = useState(false)

  useEffect(() => {
    sante().then((s) => setEtat(s.etat)).catch(() => setEtat('arrete'))
    vitrine().then((v) => setChiffres(v.chiffres)).catch(() => setChiffres([]))
  }, [])

  async function envoyer(e: FormEvent) {
    e.preventDefault()
    if (!question.trim()) return
    setAttente(true)
    setErreur('')
    try {
      setReponse(await demander(question))
    } catch {
      setReponse(null)
      setErreur("L'assistant n'a pas pu répondre. Réessaie dans un instant.")
    } finally {
      setAttente(false)
    }
  }

  return (
    <main>
      <header>
        <Logo app="rag-rgpd" />
        <BadgeLigne ligne="ai" />
        <Etat etat={etat} />
      </header>

      <form onSubmit={envoyer}>
        <label htmlFor="question">Ta question sur la protection des données</label>
        <textarea id="question" rows={3} maxLength={2000} value={question}
          onChange={(e) => setQuestion(e.target.value)} />
        <Bouton type="submit" disabled={attente}>
          {attente ? 'Recherche…' : 'Demander'}
        </Bouton>
      </form>

      {erreur && <p role="alert">{erreur}</p>}

      {reponse && (
        <section aria-live="polite">
          <p>{reponse.texte}</p>
          {reponse.refus && reponse.raison && <p className="discret">Raison du refus : {reponse.raison}.</p>}
          {reponse.sources.length > 0 && (
            <ol className="sources">
              {reponse.sources.map((s) => (
                <li key={s.numero}>
                  <a href={s.url}>{s.titre}</a>
                  {s.pages && <span>, {s.pages}</span>}
                  {s.verifie && <> <Valide>citation vérifiée</Valide></>}
                  {s.version && <span className="discret"> · texte du {s.version}</span>}
                  {s.extraits.map((x) => <blockquote key={x}>{x}</blockquote>)}
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      <footer>
        <p className="discret">{AVERTISSEMENT}</p>
        {chiffres.length > 0 && (
          <ul className="discret">
            {chiffres.map((c) => <li key={c.libelle}>{c.libelle} : {c.valeur}</li>)}
          </ul>
        )}
      </footer>
    </main>
  )
}
