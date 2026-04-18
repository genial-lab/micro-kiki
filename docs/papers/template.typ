// Academic paper template for micro-kiki papers (pandoc-compatible)
#let horizontalrule = line(length: 100%, stroke: 0.5pt + gray)
#let blockquote(content) = block(
  inset: (left: 1em, top: 0.5em, bottom: 0.5em),
  stroke: (left: 2pt + gray),
  content
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1 / 1",
)
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { pagebreak(weak: true); block(above: 1.5em, below: 1em, text(size: 16pt, weight: "bold", it.body)) }
#show heading.where(level: 2): it => block(above: 1.2em, below: 0.6em, text(size: 12pt, weight: "bold", it.body))
#show heading.where(level: 3): it => block(above: 1em, below: 0.4em, text(size: 11pt, weight: "bold", style: "italic", it.body))
#show link: set text(fill: blue.darken(20%))
#show raw.where(block: true): set block(fill: luma(240), inset: 0.8em, radius: 4pt)
