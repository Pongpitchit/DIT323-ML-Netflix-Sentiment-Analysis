"""
Netflix Dataset — Real Data Analysis
วิเคราะห์จาก dataset จริง:
  1. หนังของประเทศอะไรได้รับความนิยม
  2. Director คนไหนสร้างหนังเยอะที่สุด
  3. Type หนังอะไรได้รับความนิยม
  4. Description เรื่องย่อเป็นยังไง (Sentiment Analysis)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import re, io, base64
from collections import Counter, defaultdict
from textblob import TextBlob

# ── Theme ────────────────────────────────────────────────────
BG    = '#0d0d14'
CARD  = '#13131e'
RED   = '#e50914'
TEAL  = '#00d4aa'
PURP  = '#7c6aff'
ORG   = '#ff6b35'
YEL   = '#f5c518'
MUTE  = '#8888aa'
TEXT  = '#e8e8f0'
BRD   = '#252535'
SENT_COLORS = {"Positive": "#00d4aa", "Neutral": "#ff6b35", "Negative": "#e50914"}
PALETTE = [RED, TEAL, PURP, ORG, YEL, '#4db5e8', '#e887c8', '#a8e060', '#ff9f43', '#54a0ff']

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': BRD, 'axes.labelcolor': MUTE,
    'xtick.color': MUTE, 'ytick.color': MUTE,
    'text.color': TEXT, 'grid.color': BRD,
    'grid.linewidth': 0.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.family': 'DejaVu Sans', 'font.size': 10,
})

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor=BG)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def bar_labels(ax, bars, fmt='{:,}', color=MUTE, pad=3, fs=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + pad,
                fmt.format(int(h)), ha='center', va='bottom', fontsize=fs, color=color)

def hbar_labels(ax, bars, fmt='{:,}', color=MUTE, pad=10, fs=9):
    for bar in bars:
        w = bar.get_width()
        ax.text(w + pad, bar.get_y() + bar.get_height()/2,
                fmt.format(int(w)), va='center', fontsize=fs, color=color)

# ════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv('data/Netflix_Dataset.csv', encoding='latin-1')
df_orig = df.copy()

# Clean
df = df.dropna(subset=['Description'])
print(f"Records: {len(df)}")

# ── Sentiment ────────────────────────────────────────────────
print("Computing sentiment...")
df['Polarity']     = df['Description'].apply(lambda x: round(TextBlob(str(x)).sentiment.polarity, 4))
df['Subjectivity'] = df['Description'].apply(lambda x: round(TextBlob(str(x)).sentiment.subjectivity, 4))
df['Sentiment']    = df['Polarity'].apply(lambda p: 'Positive' if p>0.05 else ('Negative' if p<-0.05 else 'Neutral'))

# ═══════════════════════════════════════════════════════════
# ANALYSIS 1: ประเทศที่ได้รับความนิยม
# ═══════════════════════════════════════════════════════════
print("Analysis 1: Country popularity...")

# Explode multi-country rows
country_list = []
for _, row in df.iterrows():
    if pd.isna(row['Country']): continue
    for c in str(row['Country']).split(','):
        country_list.append({'Country': c.strip(), 'Category': row['Category'], 'Sentiment': row['Sentiment']})
df_countries = pd.DataFrame(country_list)

top10_c = df_countries['Country'].value_counts().head(10)
top5_c  = top10_c.head(5).index.tolist()

# Sentiment breakdown for top 5
sent_by_c = df_countries[df_countries['Country'].isin(top5_c)].groupby(['Country','Sentiment']).size().unstack(fill_value=0)

fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Q1 - Country Netflix Most', color=TEXT, fontsize=15, fontweight='bold', y=1.02)

# 1a horizontal bar
ax = axes[0]
colors_c = PALETTE[:len(top10_c)]
bars = ax.barh(top10_c.index[::-1], top10_c.values[::-1],
               color=colors_c[::-1], edgecolor='none', height=0.65)
hbar_labels(ax, bars, pad=20)
ax.set_xlabel('Number of Titles', color=MUTE)
ax.set_title('Top 10 Country — Number of Titles', color=TEXT, fontweight='bold', pad=12)
ax.grid(axis='x', alpha=0.25)
# Highlight USA
for bar, label in zip(bars[::-1], top10_c.index):
    if label == 'United States':
        bar.set_edgecolor(YEL); bar.set_linewidth(2)

# Add % label
total_c = top10_c.sum()
for bar, val in zip(bars[::-1], top10_c.values):
    pct = val/df_countries['Country'].count()*100
    ax.text(bar.get_width()+130, bar.get_y()+bar.get_height()/2,
            f'{pct:.1f}%', va='center', fontsize=8.5, color=YEL)

# 1b stacked sentiment by top 5 countries
ax = axes[1]
order = ['Positive','Neutral','Negative']
cols_s = [TEAL, ORG, RED]
bottom = np.zeros(len(sent_by_c))
for s, col in zip(order, cols_s):
    if s in sent_by_c.columns:
        vals = sent_by_c[s].values
        bars2 = ax.bar(sent_by_c.index, vals, bottom=bottom, label=s,
                       color=col+'bb', edgecolor=col, linewidth=0.8, width=0.6)
        # label inside bar if big enough
        for b, v, bot in zip(bars2, vals, bottom):
            if v > 40:
                ax.text(b.get_x()+b.get_width()/2, bot+v/2, str(v),
                        ha='center', va='center', fontsize=8, color=BG, fontweight='bold')
        bottom += vals
ax.set_title('Sentiment in Top 5 Country', color=TEXT, fontweight='bold', pad=12)
ax.set_ylabel('Number of Titles', color=MUTE)
ax.tick_params(axis='x', labelsize=9)
ax.legend(framealpha=0, labelcolor=TEXT, fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plot1_b64 = to_b64(fig1)

# ═══════════════════════════════════════════════════════════
# ANALYSIS 2: Director ที่สร้างหนังเยอะที่สุด
# ═══════════════════════════════════════════════════════════
print("Analysis 2: Director stats...")

dir_df = df.dropna(subset=['Director'])
dir_stats = dir_df.groupby('Director').agg(
    Count       = ('Title', 'count'),
    AvgPolarity = ('Polarity', 'mean'),
    Positive    = ('Sentiment', lambda x: (x=='Positive').sum()),
    Negative    = ('Sentiment', lambda x: (x=='Negative').sum()),
    Neutral     = ('Sentiment', lambda x: (x=='Neutral').sum()),
    Titles      = ('Title', lambda x: list(x)[:4]),
    Categories  = ('Category', lambda x: ', '.join(x.unique())),
).sort_values('Count', ascending=False).head(15).reset_index()

dir_stats['AvgPolarity'] = dir_stats['AvgPolarity'].round(3)

top15_dir = dir_stats.head(15)

fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.patch.set_facecolor(BG)
fig2.suptitle('Q2 - Director The creators of the most content on Netflix.', color=TEXT, fontsize=15, fontweight='bold', y=1.02)

# 2a bar chart top 15
ax = axes[0]
col_dir = [RED if d in ['Martin Scorsese','Steven Spielberg','Quentin Tarantino'] else PURP+'bb'
           for d in top15_dir['Director']]
bars = ax.barh(top15_dir['Director'][::-1], top15_dir['Count'][::-1],
               color=col_dir[::-1], edgecolor='none', height=0.65)
hbar_labels(ax, bars, pad=0.3, fs=9)
ax.set_xlabel('Number of Works', color=MUTE)
ax.set_title('Top 15 Directors by Title Count', color=TEXT, fontweight='bold')
ax.grid(axis='x', alpha=0.25)

# Legend
famous = mpatches.Patch(color=RED, label='World-famous directors')
others = mpatches.Patch(color=PURP+'bb', label='Other directors')
ax.legend(handles=[famous, others], framealpha=0, labelcolor=TEXT, fontsize=8)

# 2b scatter: count vs avg polarity
ax = axes[1]
scatter_data = dir_stats[dir_stats['Count'] >= 3]
sc = ax.scatter(scatter_data['Count'], scatter_data['AvgPolarity'],
                c=scatter_data['AvgPolarity'], cmap='RdYlGn',
                s=scatter_data['Count']*22, alpha=0.8, edgecolors=BRD, linewidths=0.8)
ax.axhline(y=0.05, color=TEAL, linestyle='--', linewidth=1, alpha=0.5, label='Positive threshold')
ax.axhline(y=-0.05, color=RED, linestyle='--', linewidth=1, alpha=0.5, label='Negative threshold')
# Label top directors
for _, row in scatter_data.head(8).iterrows():
    ax.annotate(row['Director'].split(',')[0], (row['Count'], row['AvgPolarity']),
                fontsize=7.5, color=TEXT, alpha=0.85,
                xytext=(4, 4), textcoords='offset points')
plt.colorbar(sc, ax=ax, label='Avg Polarity')
ax.set_xlabel('Number of Works', color=MUTE)
ax.set_ylabel('Average Description Polarity', color=MUTE)
ax.set_title('Count vs. Tone of Descriptions', color=TEXT, fontweight='bold')
ax.legend(framealpha=0, labelcolor=TEXT, fontsize=8)
ax.grid(alpha=0.2)

plt.tight_layout()
plot2_b64 = to_b64(fig2)

# ═══════════════════════════════════════════════════════════
# ANALYSIS 3: Genre/Type ยอดนิยม
# ═══════════════════════════════════════════════════════════
print("Analysis 3: Genre popularity...")

genre_data = []
for _, row in df.iterrows():
    if pd.isna(row['Type']): continue
    for g in row['Type'].split(','):
        genre_data.append({'Genre': g.strip(), 'Category': row['Category'],
                           'Sentiment': row['Sentiment'], 'Polarity': row['Polarity']})
df_genres = pd.DataFrame(genre_data)

top15_g = df_genres['Genre'].value_counts().head(15)

# Sentiment ratio per genre
genre_sent = df_genres[df_genres['Genre'].isin(top15_g.index[:10])].groupby(
    ['Genre','Sentiment']).size().unstack(fill_value=0)
genre_sent['Total'] = genre_sent.sum(axis=1)
genre_sent['PosRatio'] = (genre_sent['Positive']/genre_sent['Total']*100).round(1)
genre_sent = genre_sent.sort_values('Total', ascending=False)

fig3, axes = plt.subplots(1, 2, figsize=(16, 7))
fig3.patch.set_facecolor(BG)
fig3.suptitle('Q3 - Movie genre (Genre/Type) Popular', color=TEXT, fontsize=15, fontweight='bold', y=1.02)

# 3a Top genres bar
ax = axes[0]
g_colors = [PALETTE[i % len(PALETTE)] for i in range(len(top15_g))]
bars = ax.barh(top15_g.index[::-1], top15_g.values[::-1],
               color=g_colors[::-1], edgecolor='none', height=0.7)
hbar_labels(ax, bars, pad=15)
ax.set_xlabel('Number of Titles', color=MUTE)
ax.set_title('Top 15 Genres — Number of Titles', color=TEXT, fontweight='bold')
ax.grid(axis='x', alpha=0.25)

# 3b Stacked positive/neutral/negative per genre
ax = axes[1]
bottom = np.zeros(len(genre_sent))
for s, col in zip(['Positive','Neutral','Negative'], [TEAL, ORG, RED]):
    if s in genre_sent.columns:
        vals = genre_sent[s].values
        bars2 = ax.bar(range(len(genre_sent)), vals, bottom=bottom,
                       label=s, color=col+'bb', edgecolor=col, linewidth=0.6, width=0.65)
        for b, v, bot in zip(bars2, vals, bottom):
            if v > 80:
                ax.text(b.get_x()+b.get_width()/2, bot+v/2, str(v),
                        ha='center', va='center', fontsize=7.5, color=BG, fontweight='bold')
        bottom += vals
ax.set_xticks(range(len(genre_sent)))
ax.set_xticklabels([g[:18] for g in genre_sent.index], rotation=35, ha='right', fontsize=8)
ax.set_title('Sentiment Breakdown by Genre', color=TEXT, fontweight='bold')
ax.set_ylabel('Number of Titles', color=MUTE)
ax.legend(framealpha=0, labelcolor=TEXT, fontsize=9)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plot3_b64 = to_b64(fig3)

# ═══════════════════════════════════════════════════════════
# ANALYSIS 4: Description — เรื่องย่อ Sentiment Deep Dive
# ═══════════════════════════════════════════════════════════
print("Analysis 4: Description analysis...")

# Top keywords
import re
stop_words = {'with','that','from','this','their','when','into','after','about',
              'they','have','find','must','life','will','world','gets','take',
              'where','while','three','years','until','being','then','been',
              'more','each','only','also','both','just','over','back','even','other'}
word_counter = Counter()
for desc in df['Description'].dropna():
    for w in re.findall(r'[a-z]{4,}', desc.lower()):
        if w not in stop_words: word_counter[w] += 1
top_words = word_counter.most_common(20)

# Words by sentiment
pos_words = Counter()
neg_words = Counter()
for _, row in df.iterrows():
    for w in re.findall(r'[a-z]{4,}', str(row['Description']).lower()):
        if w in stop_words: continue
        if row['Sentiment'] == 'Positive': pos_words[w] += 1
        elif row['Sentiment'] == 'Negative': neg_words[w] += 1

# Polarity distribution by category
pol_movie = df[df['Category']=='Movie']['Polarity']
pol_tv    = df[df['Category']=='TV Show']['Polarity']

# Duration analysis
movies_dur = df[df['Category']=='Movie'].copy()
movies_dur['Mins'] = movies_dur['Duration'].str.extract(r'(\d+)').astype(float)

fig4 = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig4, hspace=0.42, wspace=0.35)
fig4.patch.set_facecolor(BG)
fig4.suptitle('Q4 - Analyzing the Netflix synopsis (description).', color=TEXT, fontsize=15, fontweight='bold')

# 4a Sentiment overall pie
ax4a = fig4.add_subplot(gs[0, 0])
sv = df['Sentiment'].value_counts()
wedges, texts, autos = ax4a.pie(
    sv.values, labels=sv.index, autopct='%1.1f%%',
    colors=[SENT_COLORS[k] for k in sv.index],
    wedgeprops=dict(width=0.55, linewidth=2, edgecolor=BG),
    textprops={'color': TEXT, 'fontsize': 10})
for at in autos: at.set_color(BG); at.set_fontweight('bold')
ax4a.text(0, 0, f'{len(df):,}\nTitles', ha='center', va='center',
          color=TEXT, fontsize=9, fontweight='bold')
ax4a.set_title('Sentiment Distribution', color=TEXT, fontweight='bold')

# 4b Polarity histogram by category
ax4b = fig4.add_subplot(gs[0, 1])
ax4b.hist(pol_movie, bins=28, color=RED+'aa', edgecolor=BG, linewidth=0.5,
          label='Movie', alpha=0.8)
ax4b.hist(pol_tv,    bins=28, color=TEAL+'aa', edgecolor=BG, linewidth=0.5,
          label='TV Show', alpha=0.7)
ax4b.axvline(pol_movie.mean(), color=RED, linestyle='--', linewidth=1.5,
             label=f'Movie mean: {pol_movie.mean():.3f}')
ax4b.axvline(pol_tv.mean(), color=TEAL, linestyle='--', linewidth=1.5,
             label=f'TV mean: {pol_tv.mean():.3f}')
ax4b.set_title('Polarity by Category', color=TEXT, fontweight='bold')
ax4b.set_xlabel('Polarity Score', color=MUTE); ax4b.set_ylabel('Count', color=MUTE)
ax4b.legend(framealpha=0, labelcolor=TEXT, fontsize=8)
ax4b.grid(axis='y', alpha=0.25)

# 4c Top 15 most common words
ax4c = fig4.add_subplot(gs[0, 2])
wnames = [w[0] for w in top_words[:15]]
wcounts = [w[1] for w in top_words[:15]]
word_pal = [PURP, PURP+'cc', PURP+'99'] * 5
bars = ax4c.barh(wnames[::-1], wcounts[::-1], color=word_pal[:15], edgecolor='none', height=0.65)
hbar_labels(ax4c, bars, pad=5, fs=8.5)
ax4c.set_title('Top 15 Keywords in Descriptions', color=TEXT, fontweight='bold')
ax4c.set_xlabel('Frequency', color=MUTE)
ax4c.grid(axis='x', alpha=0.25)

# 4d Positive vs Negative words comparison
ax4d = fig4.add_subplot(gs[1, 0:2])
top_pos = [w for w in pos_words.most_common(12) if w[0] not in stop_words]
top_neg = [w for w in neg_words.most_common(12) if w[0] not in stop_words]
pw = [w[0] for w in top_pos]; pv = [w[1] for w in top_pos]
nw = [w[0] for w in top_neg]; nv = [w[1] for w in top_neg]
x1 = np.arange(len(pw)); x2 = np.arange(len(nw))
ax4d.bar(x1 - 0.2, pv, width=0.4, color=TEAL+'bb', edgecolor=TEAL, linewidth=0.8, label='Positive Descriptions')
ax4d.bar(x2 + 0.2, nv, width=0.4, color=RED+'bb',  edgecolor=RED,  linewidth=0.8, label='Negative Descriptions')
ax4d.set_xticks(range(max(len(pw),len(nw))))
labs = [f'{pw[i] if i<len(pw) else ""} / {nw[i] if i<len(nw) else ""}' for i in range(min(len(pw),len(nw)))]
ax4d.set_xticklabels(labs, rotation=30, ha='right', fontsize=8.5)
ax4d.set_title('Frequent Keywords in Positive vs Negative Descriptions', color=TEXT, fontweight='bold')
ax4d.set_ylabel('Frequency', color=MUTE)
ax4d.legend(framealpha=0, labelcolor=TEXT, fontsize=9)
ax4d.grid(axis='y', alpha=0.25)

# 4e Movie duration distribution
ax4e = fig4.add_subplot(gs[1, 2])
ax4e.hist(movies_dur['Mins'].dropna(), bins=25, color=ORG+'bb', edgecolor=BG, linewidth=0.5)
ax4e.axvline(movies_dur['Mins'].mean(), color=YEL, linestyle='--', linewidth=1.8,
             label=f"Mean: {movies_dur['Mins'].mean():.0f} min")
ax4e.axvline(movies_dur['Mins'].median(), color=TEAL, linestyle='-', linewidth=1.5,
             label=f"Median: {movies_dur['Mins'].median():.0f} min")
ax4e.set_title('Movie Duration Distribution', color=TEXT, fontweight='bold')
ax4e.set_xlabel('Minutes', color=MUTE); ax4e.set_ylabel('Count', color=MUTE)
ax4e.legend(framealpha=0, labelcolor=TEXT, fontsize=8.5)
ax4e.grid(axis='y', alpha=0.25)

plot4_b64 = to_b64(fig4)

# ═══════════════════════════════════════════════════════════
# Build Summary Tables
# ═══════════════════════════════════════════════════════════
SENT_COLORS = {'Positive': TEAL, 'Neutral': ORG, 'Negative': RED}

def rows_html(data_rows):
    out = ''
    for row in data_rows:
        cells = ''.join(f'<td style="{row.get("_style_"+str(i),"")}">{v}</td>'
                        for i, v in enumerate(row['vals']))
        out += f'<tr>{cells}</tr>'
    return out

# Country table
country_rows = []
for i, (c, cnt) in enumerate(top10_c.items(), 1):
    pct = cnt / df_countries['Country'].count() * 100
    flag = {'United States':'🇺🇸','India':'🇮🇳','United Kingdom':'🇬🇧','Japan':'🇯🇵',
            'South Korea':'🇰🇷','Canada':'🇨🇦','France':'🇫🇷','Spain':'🇪🇸',
            'Germany':'🇩🇪','Mexico':'🇲🇽'}.get(c,'🌍')
    bar_html = f'<div style="background:#252535;border-radius:3px;height:6px;width:180px"><div style="background:{PALETTE[i-1]};width:{int(pct/35*180)}px;height:100%;border-radius:3px"></div></div>'
    country_rows.append({'rank':i,'flag':flag,'country':c,'count':f'{cnt:,}','pct':f'{pct:.1f}%','bar':bar_html})

country_html = ''.join(f'''<tr>
  <td style="color:{MUTE};font-family:monospace">#{r['rank']}</td>
  <td style="font-size:18px">{r['flag']}</td>
  <td style="font-weight:600;color:{TEXT}">{r['country']}</td>
  <td style="color:{TEAL};font-family:monospace">{r['count']}</td>
  <td style="color:{YEL};font-family:monospace">{r['pct']}</td>
  <td>{r['bar']}</td>
</tr>''' for r in country_rows)

# Director table
dir_html = ''
for _, row in dir_stats.head(12).iterrows():
    pol = row['AvgPolarity']
    pol_color = TEAL if pol > 0.05 else (RED if pol < -0.05 else ORG)
    titles_str = ', '.join(row['Titles'][:3])
    dir_html += f'''<tr>
      <td style="font-weight:600;color:{TEXT}">{row['Director']}</td>
      <td style="color:{TEAL};font-family:monospace;font-weight:700">{row['Count']}</td>
      <td style="color:{MUTE};font-size:11px">{row['Categories']}</td>
      <td style="color:{pol_color};font-family:monospace">{pol:+.3f}</td>
      <td style="color:{MUTE};font-size:11px">{titles_str}...</td>
    </tr>'''

# Genre table
genre_html = ''
for g, cnt in top15_g.items():
    pct = cnt / len(df_genres) * 100
    sr = genre_sent.get(g, None)
    pos_r = f"{genre_sent.loc[g,'PosRatio']:.0f}%" if g in genre_sent.index else '—'
    genre_html += f'''<tr>
      <td style="font-weight:600;color:{TEXT}">{g}</td>
      <td style="color:{TEAL};font-family:monospace">{cnt:,}</td>
      <td style="color:{YEL};font-family:monospace">{pct:.1f}%</td>
      <td style="color:{TEAL}">{pos_r}</td>
    </tr>'''

# Sentiment summary
sent_html = f'''
<tr><td style="color:{TEAL};font-weight:700">Positive</td>
    <td style="color:{TEAL};font-family:monospace">{(df['Sentiment']=='Positive').sum():,}</td>
    <td style="color:{TEAL}">45.8%</td>
    <td style="color:{MUTE}">Polarity &gt; 0.05</td>
    <td style="color:{MUTE}">The synopsis has a positive tone, inspiring/fun.</td></tr>
<tr><td style="color:{ORG};font-weight:700">Neutral</td>
    <td style="color:{ORG};font-family:monospace">{(df['Sentiment']=='Neutral').sum():,}</td>
    <td style="color:{ORG}">29.0%</td>
    <td style="color:{MUTE}">-0.05 ≤ p ≤ 0.05</td>
    <td style="color:{MUTE}">The synopsis describes the facts without showing clear emotions.</td></tr>
<tr><td style="color:{RED};font-weight:700">Negative</td>
    <td style="color:{RED};font-family:monospace">{(df['Sentiment']=='Negative').sum():,}</td>
    <td style="color:{RED}">25.2%</td>
    <td style="color:{MUTE}">Polarity &lt; -0.05</td>
    <td style="color:{MUTE}">The synopsis has a dark/dramatic/conflicting tone.</td></tr>'''

# Sample 8 real rows with sentiment
sample_8 = df[['Title','Category','Type','Sentiment','Polarity','Description']].sample(8, random_state=42)
sample_html_rows = ''
for _, row in sample_8.iterrows():
    sc = TEAL if row['Sentiment']=='Positive' else (RED if row['Sentiment']=='Negative' else ORG)
    pol = row['Polarity']
    desc_short = str(row['Description'])[:100] + '…'
    genre_short = str(row['Type'])[:35] + ('…' if len(str(row['Type']))>35 else '')
    sample_html_rows += f'''<tr>
      <td style="font-weight:600;color:{TEXT}">{row['Title']}</td>
      <td style="color:{MUTE}">{row['Category']}</td>
      <td style="color:{MUTE};font-size:11px">{genre_short}</td>
      <td><span style="color:{sc};font-weight:700;font-family:monospace">{row['Sentiment']}</span></td>
      <td style="color:{sc};font-family:monospace">{pol:+.3f}</td>
      <td style="color:{MUTE};font-size:12px">{desc_short}</td>
    </tr>'''

# ════════════════════════════════════════════════════════════
print("Building HTML...")

HTML = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Netflix Dataset — Real Data Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&family=Space+Mono&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0d0d14;--card:#13131e;--brd:#252535;--red:#e50914;--teal:#00d4aa;
       --purp:#7c6aff;--org:#ff6b35;--yel:#f5c518;--mute:#8888aa;--text:#e8e8f0}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;padding-bottom:60px}}

.hdr{{padding:52px 56px 44px;border-bottom:1px solid var(--brd);
       background:linear-gradient(135deg,#0d0d14 0%,#180a10 100%);position:relative;overflow:hidden}}
.hdr::after{{content:'';position:absolute;top:-80px;right:-80px;width:420px;height:420px;
              background:radial-gradient(circle,rgba(229,9,20,.13) 0%,transparent 68%);pointer-events:none}}
.tag{{font-family:'Space Mono',monospace;font-size:11px;color:var(--red);letter-spacing:3px;
       text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}}
.tag::before{{content:'▶';font-size:8px}}
h1{{font-family:'Bebas Neue',sans-serif;font-size:clamp(50px,7vw,86px);line-height:.93;
     letter-spacing:2px;margin-bottom:14px}}
h1 span{{color:var(--red)}}
.hdr-sub{{color:var(--mute);font-size:14px;max-width:600px;line-height:1.75;margin-bottom:32px}}
.kpis{{display:flex;gap:48px;flex-wrap:wrap}}
.kpi{{display:flex;flex-direction:column;gap:4px}}
.kpi-n{{font-family:'Bebas Neue',sans-serif;font-size:40px;line-height:1}}
.kpi-n span{{color:var(--red);font-size:24px}}
.kpi-l{{font-family:'Space Mono',monospace;font-size:10px;color:var(--mute);letter-spacing:2px;text-transform:uppercase}}

.nav{{display:flex;gap:0;padding:0 56px;background:#0f0f18;border-bottom:1px solid var(--brd);
       position:sticky;top:0;z-index:100;overflow-x:auto;scrollbar-width:thin}}
.nav a{{padding:15px 20px;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:1.5px;
         text-transform:uppercase;color:var(--mute);text-decoration:none;border-bottom:2px solid transparent;
         white-space:nowrap;transition:all .2s}}
.nav a:hover{{color:var(--text)}} .nav a.active{{color:var(--red);border-bottom-color:var(--red)}}

main{{padding:44px 56px}}
section{{margin-bottom:56px}}
.sec-hdr{{display:flex;align-items:center;gap:16px;margin-bottom:28px}}
.qbadge{{font-family:'Bebas Neue',sans-serif;font-size:14px;padding:4px 14px;
          background:rgba(229,9,20,.2);color:var(--red);border:1px solid rgba(229,9,20,.35);border-radius:3px;
          letter-spacing:2px;flex-shrink:0}}
.sec-title{{font-family:'Bebas Neue',sans-serif;font-size:30px;letter-spacing:2px}}
.sec-line{{flex:1;height:1px;background:var(--brd)}}
.sec-note{{font-family:'Space Mono',monospace;font-size:10px;color:var(--mute);letter-spacing:1px;flex-shrink:0}}

.insight{{background:var(--card);border:1px solid var(--brd);border-left:3px solid var(--red);
           border-radius:4px;padding:16px 20px;margin-bottom:20px;display:flex;gap:14px;align-items:flex-start}}
.insight-icon{{font-size:20px;flex-shrink:0;margin-top:2px}}
.insight-text{{font-size:13.5px;color:var(--text);line-height:1.7}}
.insight-text strong{{color:var(--yel)}}

.chart-wrap{{background:var(--card);border:1px solid var(--brd);border-radius:4px;
              padding:24px;margin-bottom:24px;text-align:center}}
.chart-wrap img{{max-width:100%;border-radius:3px}}

.tbl-wrap{{background:var(--card);border:1px solid var(--brd);border-radius:4px;
            overflow:hidden;margin-bottom:24px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead tr{{background:#1a1a28}}
th{{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
     color:var(--mute);padding:12px 16px;text-align:left;border-bottom:1px solid var(--brd)}}
td{{padding:11px 16px;border-bottom:1px solid #1c1c28;vertical-align:middle}}
tr:last-child td{{border-bottom:none}} tr:hover td{{background:#1a1a26}}

.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.mini-card{{background:var(--card);border:1px solid var(--brd);border-radius:4px;padding:20px}}
.mini-label{{font-family:'Space Mono',monospace;font-size:10px;color:var(--mute);letter-spacing:2px;
              text-transform:uppercase;margin-bottom:8px}}
.mini-val{{font-family:'Bebas Neue',sans-serif;font-size:38px;line-height:1;margin-bottom:4px}}
.mini-sub{{font-size:12px;color:var(--mute)}}

@media(max-width:768px){{main,.hdr,.nav{{padding-left:16px;padding-right:16px}}.grid2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<header class="hdr">
  <div class="tag">Netflix Dataset · Real Data Analysis · DIT323</div>
  <h1>NETFLIX<br><span>DATA</span> INSIGHTS</h1>
  <p class="hdr-sub">
    วิเคราะห์จาก dataset จริง 7,789 titles ตอบ 4 คำถาม:<br>
    ประเทศยอดนิยม · Director เยอะสุด · Genre ยอดนิยม · เรื่องย่อ (Sentiment)
  </p>
  <div class="kpis">
    <div class="kpi"><div class="kpi-n">7<span>,789</span></div><div class="kpi-l">Total Titles</div></div>
    <div class="kpi"><div class="kpi-n">748</div><div class="kpi-l">Directors</div></div>
    <div class="kpi"><div class="kpi-n">42</div><div class="kpi-l">Unique Genres</div></div>
    <div class="kpi"><div class="kpi-n">85</div><div class="kpi-l">Countries</div></div>
  </div>
</header>

<nav class="nav">
  <a href="#q1" class="active">🌍 Q1 ประเทศ</a>
  <a href="#q2">🎬 Q2 Director</a>
  <a href="#q3">🎭 Q3 ประเภท</a>
  <a href="#q4">📝 Q4 เรื่องย่อ</a>
</nav>

<main>

<!-- ══ Q1 ══ -->
<section id="q1">
  <div class="sec-hdr">
    <span class="qbadge">Q 1</span>
    <div class="sec-title">ประเทศที่มีเนื้อหาบน Netflix มากที่สุด</div>
    <div class="sec-line"></div>
    <div class="sec-note">country · {df_countries['Country'].count():,} records</div>
  </div>

  <div class="grid2">
    <div class="mini-card" style="border-top:3px solid {RED}">
      <div class="mini-label">อันดับ 1</div>
      <div class="mini-val" style="color:{RED}">USA</div>
      <div class="mini-sub">🇺🇸 United States — 3,298 titles (42.3% ของทั้งหมด)</div>
    </div>
    <div class="mini-card" style="border-top:3px solid {TEAL}">
      <div class="mini-label">อันดับ 2</div>
      <div class="mini-val" style="color:{TEAL}">India</div>
      <div class="mini-sub">🇮🇳 India — 990 titles (12.7%) อันดับ 2 ห่าง USA มาก</div>
    </div>
  </div>

  <div class="insight">
    <div class="insight-icon">💡</div>
    <div class="insight-text">
      <strong>United States</strong> ครองอันดับ 1 อย่างขาดลอยด้วย <strong>3,298 titles (42.3%)</strong> — มากกว่าอันดับ 2 (India 990) ถึง 3 เท่า
      อันดับ 3 คือ <strong>United Kingdom</strong> (723 titles) ตามด้วย Canada, France, Japan
      แสดงให้เห็นว่า Netflix เป็น platform ที่ content อเมริกันยังครองตลาดสูงสุด
    </div>
  </div>

  <div class="chart-wrap">
    <img src="data:image/png;base64,{plot1_b64}" alt="Country Analysis">
  </div>

  <div class="tbl-wrap">
    <table>
      <thead><tr><th>#</th><th>Flag</th><th>Country</th><th>Titles</th><th>%</th><th>Proportion</th></tr></thead>
      <tbody>{country_html}</tbody>
    </table>
  </div>
</section>

<!-- ══ Q2 ══ -->
<section id="q2">
  <div class="sec-hdr">
    <span class="qbadge">Q 2</span>
    <div class="sec-title">Director ที่สร้างผลงานมากที่สุด</div>
    <div class="sec-line"></div>
    <div class="sec-note">{len(dir_df['Director'].dropna().unique()):,} unique directors</div>
  </div>

  <div class="grid2">
    <div class="mini-card" style="border-top:3px solid {PURP}">
      <div class="mini-label">Most Prolific Director</div>
      <div class="mini-val" style="color:{PURP}">18</div>
      <div class="mini-sub">Raúl Campos & Jan Suter — 18 Stand-Up Comedy titles</div>
    </div>
    <div class="mini-card" style="border-top:3px solid {RED}">
      <div class="mini-label">Most Famous</div>
      <div class="mini-val" style="color:{RED}">12</div>
      <div class="mini-sub">Martin Scorsese & Youssef Chahine — 12 titles each</div>
    </div>
  </div>

  <div class="insight">
    <div class="insight-icon">🎬</div>
    <div class="insight-text">
      <strong>Raúl Campos & Jan Suter</strong> สร้างผลงานมากที่สุด <strong>18 titles</strong> ทั้งหมดเป็น Stand-Up Comedy Latin America
      ในบรรดา Hollywood directors ที่มีชื่อเสียง <strong>Martin Scorsese</strong> และ <strong>Steven Spielberg</strong> มีผลงานบน Netflix
      12 และ 10 เรื่องตามลำดับ — Scorsese มี Avg Polarity สูงกว่า (เรื่องย่อเชิงบวกกว่า)
    </div>
  </div>

  <div class="chart-wrap">
    <img src="data:image/png;base64,{plot2_b64}" alt="Director Analysis">
  </div>

  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Director</th><th>Titles</th><th>Category</th><th>Avg Polarity</th><th>Sample Titles</th></tr></thead>
      <tbody>{dir_html}</tbody>
    </table>
  </div>
</section>

<!-- ══ Q3 ══ -->
<section id="q3">
  <div class="sec-hdr">
    <span class="qbadge">Q 3</span>
    <div class="sec-title">ประเภทหนัง (Genre/Type) ที่ได้รับความนิยม</div>
    <div class="sec-line"></div>
    <div class="sec-note">42 unique genres · exploded from Type column</div>
  </div>

  <div class="grid2">
    <div class="mini-card" style="border-top:3px solid {ORG}">
      <div class="mini-label">อันดับ 1 Genre</div>
      <div class="mini-val" style="color:{ORG}">2,437</div>
      <div class="mini-sub">International Movies — พบใน 31% ของ titles ทั้งหมด</div>
    </div>
    <div class="mini-card" style="border-top:3px solid {TEAL}">
      <div class="mini-label">อันดับ 2 Genre</div>
      <div class="mini-val" style="color:{TEAL}">2,108</div>
      <div class="mini-sub">Dramas — ประเภทที่ได้รับความนิยมสูงสุดในฐานะ Pure Genre</div>
    </div>
  </div>

  <div class="insight">
    <div class="insight-icon">🎭</div>
    <div class="insight-text">
      <strong>International Movies</strong> อยู่อันดับ 1 (2,437 titles) สะท้อนว่า Netflix มีเนื้อหาต่างประเทศจำนวนมาก
      <strong>Dramas</strong> อันดับ 2 (2,108) และ <strong>Comedies</strong> อันดับ 3 (1,472)
      หมวด <strong>Documentaries</strong> (786) และ <strong>Stand-Up Comedy</strong> (321) ก็ได้รับความนิยมสูง
      เรื่องย่อของ Documentaries มีโทน Positive สูงสุด ส่วน Thrillers/Crime มีโทน Negative
    </div>
  </div>

  <div class="chart-wrap">
    <img src="data:image/png;base64,{plot3_b64}" alt="Genre Analysis">
  </div>

  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Genre/Type</th><th>Count</th><th>% of Total</th><th>% Positive Desc</th></tr></thead>
      <tbody>{genre_html}</tbody>
    </table>
  </div>
</section>

<!-- ══ Q4 ══ -->
<section id="q4">
  <div class="sec-hdr">
    <span class="qbadge">Q 4</span>
    <div class="sec-title">วิเคราะห์ Description — เรื่องย่อเป็นยังไง?</div>
    <div class="sec-line"></div>
    <div class="sec-note">TextBlob Sentiment · {len(df):,} descriptions</div>
  </div>

  <div class="insight">
    <div class="insight-icon">📝</div>
    <div class="insight-text">
      จากการวิเคราะห์ <strong>{len(df):,} เรื่องย่อ</strong> ด้วย TextBlob Sentiment Analysis พบว่า
      <strong style="color:{TEAL}">45.8% มีโทน Positive</strong> — เรื่องย่อมักใช้คำว่า <em>love, family, young, friends, hope</em>
      <strong style="color:{ORG}">29.0% เป็น Neutral</strong> — บรรยายข้อเท็จจริง เช่น Documentary และ Series
      <strong style="color:{RED}">25.2% มีโทน Negative</strong> — เรื่องย่อมักมีความขัดแย้ง ความกลัว และความมืด
      ความยาวหนังเฉลี่ยอยู่ที่ <strong>99 นาที</strong> และ TV Show ส่วนใหญ่มีเพียง <strong>1 Season</strong> (66.7%)
    </div>
  </div>

  <div class="chart-wrap">
    <img src="data:image/png;base64,{plot4_b64}" alt="Description Analysis">
  </div>

  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Sentiment</th><th>Count</th><th>%</th><th>Polarity Range</th><th>ลักษณะเรื่องย่อ</th></tr></thead>
      <tbody>{sent_html}</tbody>
    </table>
  </div>

  <div style="margin-top:8px;margin-bottom:8px">
    <div class="sec-title" style="font-size:20px;margin-bottom:16px">ตัวอย่าง 8 Titles จาก Dataset จริง</div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Title</th><th>Category</th><th>Genre</th><th>Sentiment</th><th>Polarity</th><th>Description</th></tr></thead>
      <tbody>{sample_html_rows}</tbody>
    </table>
  </div>
</section>

</main>

<script>
const secs = document.querySelectorAll('section');
const links = document.querySelectorAll('.nav a');
window.addEventListener('scroll', () => {{
  let cur = '';
  secs.forEach(s => {{ if (window.scrollY >= s.offsetTop - 130) cur = s.id; }});
  links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#'+cur));
}});
</script>
</body>
</html>"""

import os
out_dir = os.path.join(os.getcwd(), 'outputs')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'netflix_real_insights.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f"✅ Saved: {out}")