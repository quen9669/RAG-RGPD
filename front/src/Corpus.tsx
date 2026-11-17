import { Component, lazy, Suspense, type ReactNode } from 'react'

const Constellation = lazy(() => import('./Constellation'))

const FAMILLES = ['RGPD (articles)', 'CNPD (brochures)', 'textes ferroviaires européens', 'open data']

const webgl = () => {
  try {
    return !!document.createElement('canvas').getContext('webgl2')
  } catch {
    return false
  }
}

/** Alternative non-3D : même information, en texte. */
function Liste({ description }: { description: string }) {
  return (
    <section aria-labelledby="corpus">
      <h2 id="corpus">Le corpus</h2>
      <p className="discret">{description}</p>
      <ul>{FAMILLES.map((f) => <li key={f}>{f}</li>)}</ul>
    </section>
  )
}

class Repli extends Component<{ repli: ReactNode; children: ReactNode }, { erreur: boolean }> {
  state = { erreur: false }
  static getDerivedStateFromError() {
    return { erreur: true }
  }
  render() {
    return this.state.erreur ? this.props.repli : this.props.children
  }
}

export function Corpus({ chiffres }: { chiffres: { libelle: string; valeur: number }[] }) {
  const description =
    'Constellation du corpus : les textes du RGPD, les brochures de la CNPD, les textes ferroviaires européens et l\'open data ' +
    'gravitent autour d\'un noyau. Dans cette démonstration, une question envoie des particules roses vers les passages cités, ' +
    'qui passent au vert avec ✓ une fois la citation vérifiée.' +
    (chiffres.length ? ' ' + chiffres.map((c) => `${c.libelle} : ${c.valeur}.`).join(' ') : '')
  const liste = <Liste description={description} />
  if (!webgl()) return liste
  return (
    <Repli repli={liste}>
      <Suspense fallback={liste}>
        <Constellation description={description} />
      </Suspense>
    </Repli>
  )
}
