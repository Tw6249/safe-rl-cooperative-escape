%% cjas_journal.cls
%% Standalone Chinese Control & Robotics Top Journal Double-Column LaTeX Template
%% Modeled after Acta Automatica Sinica (自动化学报) / Control and Decision (控制与决策)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cjas_journal}[2026/11/09 v1.0 Chinese Control Journal Double-Column Class]

\LoadClass[10pt,twocolumn,twoside,a4paper,UTF8,fontset=windows]{ctexart}

\RequirePackage{geometry}
\geometry{
  a4paper,
  top=22mm,
  bottom=21mm,
  left=16mm,
  right=16mm,
  columnsep=7mm,
  headheight=14pt,
  headsep=6mm,
  footskip=10mm
}

\RequirePackage{amsmath,amssymb,amsfonts,amsthm,mathtools,bm}
\RequirePackage{graphicx}
\RequirePackage{booktabs,tabularx,multirow,array,makecell}
\RequirePackage{enumitem}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{xcolor}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage[hypertexnames=false,colorlinks=true,linkcolor=blue!70!black,citecolor=blue!70!black,urlcolor=blue!70!black]{hyperref}

% Fonts and typography
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\setlength{\parskip}{0pt}
\linespread{1.12}

% Caption setup (bilingual / journal style)
\captionsetup{
  font=footnotesize,
  labelfont=bf,
  labelsep=quad,
  skip=4pt
}
\captionsetup[table]{position=top,skip=4pt}

% Section formatting
\ctexset{
  section = {
    format = \large\bfseries\raggedright,
    name = {,},
    number = \arabic{section},
    aftername = \hspace{0.6em},
    beforeskip = 1.6ex plus 0.4ex minus 0.2ex,
    afterskip = 1.0ex plus 0.2ex
  },
  subsection = {
    format = \normalsize\bfseries\raggedright,
    aftername = \hspace{0.5em},
    beforeskip = 1.2ex plus 0.3ex minus 0.2ex,
    afterskip = 0.7ex plus 0.2ex
  },
  subsubsection = {
    format = \normalsize\bfseries\raggedright,
    aftername = \hspace{0.5em},
    beforeskip = 0.9ex plus 0.2ex minus 0.1ex,
    afterskip = 0.5ex plus 0.1ex
  }
}

% Theorem environments
\newtheoremstyle{journalthm}
  {4pt}{4pt}{\normalfont}{0pt}{\bfseries}{.}{0.5em}{}
\theoremstyle{journalthm}
\newtheorem{theorem}{定理}
\newtheorem{proposition}{命题}
\newtheorem{lemma}{引理}
\newtheorem{corollary}{推论}
\newtheorem{assumption}{假设}
\newtheorem{definition}{定义}
\newtheorem{remark}{注}

\renewenvironment{proof}[1][证明]{\par
  \pushQED{\qed}%
  \normalfont \topsep2pt \partopsep0pt %
  \trivlist
  \item[\hskip\labelsep\bfseries #1.\hspace{0.4em}]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}

% Compact lists
\setlist[enumerate]{leftmargin=1.8em,itemsep=1pt,topsep=2pt,parsep=0pt}
\setlist[itemize]{leftmargin=1.6em,itemsep=1pt,topsep=2pt,parsep=0pt}

% Header and footer
\newcommand{\JournalName}{自动化学报 / 控制理论与应用（期刊投递排版稿）}
\newcommand{\JournalRunningTitle}{耦合安全约束下多机器人协同脱困的包络等变图强化学习与邻域近端分配控制}
\newcommand{\JournalRunningAuthor}{谭为，陈鑫}
\newcommand{\journalheader}[3]{%
  \renewcommand{\JournalName}{#1}%
  \renewcommand{\JournalRunningTitle}{#2}%
  \renewcommand{\JournalRunningAuthor}{#3}%
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE]{\footnotesize \JournalName}
\fancyhead[RO]{\footnotesize \JournalRunningAuthor：\JournalRunningTitle}
\fancyhead[LO,RE]{\footnotesize \thepage}
\renewcommand{\headrulewidth}{0.4pt}

\fancypagestyle{firstpage}{%
  \fancyhf{}
  \fancyhead[L]{\footnotesize \JournalName}
  \fancyhead[R]{\footnotesize Vol.\ XX, No.\ X\quad 2026}
  \fancyfoot[C]{\footnotesize \thepage}
  \renewcommand{\headrulewidth}{0.6pt}
}

% Metadata macros for bilingual frontmatter
\newcommand{\TitleZH}[1]{\def\@TitleZH{#1}}
\newcommand{\AuthorsZH}[1]{\def\@AuthorsZH{#1}}
\newcommand{\AffilZH}[1]{\def\@AffilZH{#1}}
\newcommand{\AbstractZH}[1]{\def\@AbstractZH{#1}}
\newcommand{\KeywordsZH}[1]{\def\@KeywordsZH{#1}}
\newcommand{\CLCCode}[1]{\def\@CLCCode{#1}}

\newcommand{\TitleEN}[1]{\def\@TitleEN{#1}}
\newcommand{\AuthorsEN}[1]{\def\@AuthorsEN{#1}}
\newcommand{\AffilEN}[1]{\def\@AffilEN{#1}}
\newcommand{\AbstractEN}[1]{\def\@AbstractEN{#1}}
\newcommand{\KeywordsEN}[1]{\def\@KeywordsEN{#1}}
\newcommand{\FirstPageFootnote}[1]{\def\@FirstPageFootnote{#1}}

\newcommand{\makejournaltitle}{%
  \twocolumn[{%
    \begin{center}
      \vspace*{1mm}
      {\LARGE\heiti \@TitleZH \par}
      \vspace{3.2mm}
      {\normalsize\kaishu \@AuthorsZH \par}
      \vspace{1.5mm}
      {\footnotesize\songti (\@AffilZH) \par}
    \end{center}
    \vspace{2.5mm}
    \noindent{\footnotesize{\heiti 摘\quad 要\quad}{\songti \@AbstractZH}\par}
    \vspace{1.5mm}
    \noindent{\footnotesize{\heiti 关键词\quad}{\songti \@KeywordsZH}\par}
    \vspace{1.0mm}
    \noindent{\footnotesize{\heiti 中图分类号\quad}{\songti \@CLCCode}\qquad{\heiti 文献标志码\quad}A\par}
    \vspace{3.5mm}
    \begin{center}
      {\large\bfseries \@TitleEN \par}
      \vspace{2.2mm}
      {\small \@AuthorsEN \par}
      \vspace{1.2mm}
      {\footnotesize\itshape (\@AffilEN) \par}
    \end{center}
    \vspace{2.0mm}
    \noindent{\footnotesize{\bfseries Abstract\quad}\@AbstractEN\par}
    \vspace{1.5mm}
    \noindent{\footnotesize{\bfseries Key words\quad}\@KeywordsEN\par}
    \vspace{4.5mm}
    \hrule height 0.5pt
    \vspace{4.5mm}
  }]%
  \thispagestyle{firstpage}%
  \begin{NoHyper}%
  \renewcommand{\thefootnote}{}%
  \footnotetext{\scriptsize \@FirstPageFootnote}%
  \renewcommand{\thefootnote}{\arabic{footnote}}%
  \end{NoHyper}%
}

% Math shortcuts consistent with unified notation
\newcommand{\bp}{\boldsymbol{p}}
\newcommand{\bu}{\boldsymbol{u}}
\newcommand{\bg}{\boldsymbol{g}}
\newcommand{\be}{\boldsymbol{e}}
\newcommand{\br}{\boldsymbol{r}}
\newcommand{\bL}{\boldsymbol{L}_{\mathrm{p}}}
\newcommand{\bJ}{\boldsymbol{J}}
\newcommand{\I}{\boldsymbol{I}}
\newcommand{\lr}{\lambda_{\mathrm{req}}}
\newcommand{\ds}{d_{\mathrm{safe}}}
\newcommand{\norm}[1]{\lVert#1\rVert}
\newcommand{\ip}[2]{\langle#1,#2\rangle}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\col}{\operatorname{col}}
\newcommand{\clip}{\operatorname{clip}}
\newcommand{\tr}{\operatorname{tr}}

\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\dbltopfraction}{0.92}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.88}
\renewcommand{\dblfloatpagefraction}{0.88}
\setcounter{topnumber}{3}
\setcounter{dbltopnumber}{3}
\setcounter{totalnumber}{4}

\allowdisplaybreaks[1]
\endinput
