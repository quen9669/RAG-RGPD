// Scène « la constellation » (EXP-03, charte/3d.md) : uniquement le kit @quen9669/lignes/scene.
import { Edges, Html, Line } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { FlowParticles, GridFloor, SceneShell, useScene } from '@quen9669/lignes/scene'
import { useMemo, useRef, useState, type MutableRefObject } from 'react'
import * as THREE from 'three'

type Famille = { nom: string; rayon: number; inclinaison: number; vitesse: number; docs: string[]; cite: number }

// Jeu de démonstration figé (3D-8) : la scène est une animation, pas une réponse réelle.
const FAMILLES: Famille[] = [
  { nom: 'RGPD', rayon: 3, inclinaison: 0.1, vitesse: 0.05, docs: ['art. 5', 'art. 6', 'art. 17', 'art. 32', 'art. 33'], cite: 2 },
  { nom: 'CNPD', rayon: 4.6, inclinaison: -0.15, vitesse: -0.04, docs: ['guide', 'fiche', 'avis', 'brochure'], cite: 1 },
  { nom: 'Rail UE', rayon: 6.2, inclinaison: 0.2, vitesse: 0.03, docs: ['dir. 2012/34', 'règl. 1371', 'dir. 2016/797', 'RGPD rail'], cite: 0 },
  { nom: 'Open data', rayon: 7.8, inclinaison: -0.1, vitesse: -0.025, docs: ['jeu 1', 'jeu 2', 'jeu 3', 'jeu 4'], cite: -1 },
]
const CYCLE = 9 // secondes : 0-2 repos, 2-5 recherche (rose), 5-9 vérifié (vert ✓)
const cle = (f: Famille, i: number) => `${f.nom}-${i}`

type Etape = 0 | 1 | 2

class Lien extends THREE.Curve<THREE.Vector3> {
  cible = new THREE.Vector3()
  getPoint(t: number, sortie = new THREE.Vector3()) {
    return sortie.copy(this.cible).multiplyScalar(t)
  }
  getPointAt(u: number, sortie?: THREE.Vector3) {
    return this.getPoint(u, sortie)
  }
}

function Plaquette({ titre, etape, citee, reg }: { titre: string; etape: Etape; citee: boolean; reg: (o: THREE.Object3D | null) => void }) {
  const { jetons } = useScene()
  const couleur = !citee || etape === 0 ? jetons.card : etape === 1 ? jetons['line-ai'] : jetons['line-dg']
  const texte = citee && etape === 2 ? `✓ ${titre}` : titre
  return (
    <group ref={reg}>
      <mesh>
        <boxGeometry args={[1.1, 0.12, 0.7]} />
        <meshStandardMaterial color={couleur} roughness={0.8} metalness={0} />
        <Edges color={jetons.border} />
      </mesh>
      <Html center position={[0, 0.35, 0]} style={{ pointerEvents: 'none' }} zIndexRange={[10, 0]}>
        <span aria-hidden="true" className="lg-mono"
          style={{ fontSize: 11, whiteSpace: 'nowrap', color: citee && etape === 2 ? jetons['line-dg'] : jetons.muted }}>
          {texte}
        </span>
      </Html>
    </group>
  )
}

function Orbite({ famille, etape, citees }: { famille: Famille; etape: Etape; citees: MutableRefObject<Map<string, THREE.Object3D>> }) {
  const { jetons } = useScene()
  const groupe = useRef<THREE.Group>(null)
  const anneau = useMemo(
    () => Array.from({ length: 65 }, (_, i) => [Math.cos((i / 64) * 2 * Math.PI) * famille.rayon, 0, Math.sin((i / 64) * 2 * Math.PI) * famille.rayon] as [number, number, number]),
    [famille.rayon],
  )
  useFrame((_, dt) => {
    if (groupe.current) groupe.current.rotation.y += dt * famille.vitesse
  })
  return (
    <group rotation={[famille.inclinaison, 0, 0]} position={[0, 1.6, 0]}>
      <group ref={groupe}>
        <Line points={anneau} color={jetons.rule} lineWidth={1} />
        {famille.docs.map((titre, i) => {
          const a = (i / famille.docs.length) * 2 * Math.PI
          const citee = i === famille.cite
          return (
            <group key={cle(famille, i)} position={[Math.cos(a) * famille.rayon, 0, Math.sin(a) * famille.rayon]}>
              <Plaquette titre={titre} etape={etape} citee={citee}
                reg={(o) => { if (citee) o ? citees.current.set(cle(famille, i), o) : citees.current.delete(cle(famille, i)) }} />
            </group>
          )
        })}
      </group>
    </group>
  )
}

/** Particules roses du noyau vers un passage cité, qui suit l'orbite. */
function Flux({ passage, actif }: { passage: () => THREE.Object3D | undefined; actif: boolean }) {
  const lien = useMemo(() => new Lien(), [])
  const groupe = useRef<THREE.Group>(null)
  useFrame(() => {
    passage()?.getWorldPosition(lien.cible)
    lien.cible.y -= 1.6
    if (groupe.current) groupe.current.visible = actif
  })
  return <group ref={groupe} position={[0, 1.6, 0]}><FlowParticles trace={lien} ligne="ai" nombre={14} vitesse={0.35} /></group>
}

function Monde() {
  const { reduit, jetons } = useScene()
  const [etape, setEtape] = useState<Etape>(reduit ? 2 : 0)
  const citees = useRef(new Map<string, THREE.Object3D>())
  useFrame(({ clock }) => {
    if (reduit) return
    const t = clock.elapsedTime % CYCLE
    const e: Etape = t < 2 ? 0 : t < 5 ? 1 : 2
    if (e !== etape) setEtape(e)
  })
  const e: Etape = reduit ? 2 : etape
  return (
    <>
      <GridFloor />
      <mesh position={[0, 1.6, 0]}>
        <icosahedronGeometry args={[0.8, 1]} />
        <meshStandardMaterial color={jetons.card} roughness={0.8} metalness={0} />
        <Edges color={jetons.border} />
      </mesh>
      {FAMILLES.map((f) => <Orbite key={f.nom} famille={f} etape={e} citees={citees} />)}
      {FAMILLES.flatMap((f) => f.cite >= 0 ? [cle(f, f.cite)] : []).map((k) => (
        <Flux key={k} passage={() => citees.current.get(k)} actif={e === 1} />
      ))}
    </>
  )
}

export default function Constellation({ description }: { description: string }) {
  return (
    <SceneShell description={description} hauteur={440} vue={[11, 9, 11]} cible={[0, 1.6, 0]} rayon={30} demo>
      <Monde />
    </SceneShell>
  )
}
