\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{templates/cjc-utf8}[2026/08/04 UTF-8 compatibility class based on the CJC submission template]

% The official CjC.cls is a GBK/CJK-era class.  This compatibility class keeps
% its A4 type area, 10.5 bp body size, 8 mm gutter, two-column body, and heading
% hierarchy while using ctex/XeLaTeX for reproducible UTF-8 compilation.
\LoadClass[UTF8,fontset=none,a4paper,twocolumn,twoside,zihao=5]{ctexart}

\RequirePackage{geometry}
\geometry{
  a4paper,
  left=19mm,
  right=19mm,
  top=31mm,
  bottom=31mm,
  headheight=6mm,
  headsep=4mm,
  footskip=12mm,
  columnsep=8mm
}

\RequirePackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{DejaVu Sans Mono}
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\setCJKfamilyfont{zhsong}{Noto Serif CJK SC}
\setCJKfamilyfont{zhhei}{Noto Sans CJK SC}
\setCJKfamilyfont{zhfs}{Noto Serif CJK SC}
\setCJKfamilyfont{zhkai}{Noto Serif CJK SC}

\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{enumitem}

\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\setlength{\columnsep}{8mm}
\linespread{1.14}
\setlength{\abovedisplayskip}{2pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{3pt plus 2pt minus 2pt}
\setlength{\abovedisplayshortskip}{2pt plus 1pt minus 1pt}
\setlength{\belowdisplayshortskip}{3pt plus 2pt minus 2pt}
\setlist{nosep,leftmargin=2em}

\setcounter{secnumdepth}{0}
\titleformat{\section}{\bfseries\zihao{4}}{}{0pt}{}
\titleformat{\subsection}{\bfseries\zihao{5}}{}{0pt}{}
\titlespacing*{\section}{0pt}{1.0em}{0.45em}
\titlespacing*{\subsection}{0pt}{0.75em}{0.3em}

\captionsetup{
  font=small,
  labelsep=quad,
  justification=centering,
  singlelinecheck=false,
  skip=4pt
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[CE]{\zihao{-5} 计\quad 算\quad 机\quad 学\quad 报（投稿稿）}
\fancyhead[CO]{\zihao{-5} 面向异构 SPN 的运行时结构先验神经区分方法}
\fancyfoot[C]{\zihao{-5}\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\AtBeginDocument{\zihao{5}}
