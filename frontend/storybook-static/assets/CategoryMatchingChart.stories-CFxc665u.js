import{C as ce}from"./CategoryMatchingChart-CjF6RzyR.js";import"./jsx-runtime-Bywvkw1S.js";import"./index-CleY8y_P.js";import"./_commonjsHelpers-Cpj98o6Y.js";import"./cn-CytzSlOG.js";import"./index-DBU18h-i.js";import"./index-BAUB8I8r.js";import"./index-CuYJjLYj.js";import"./useMediaQuery-Dj87e1Qj.js";import"./throttle-BoqV9_75.js";import"./mapValues-yqX3NJIo.js";import"./tiny-invariant-CopsF_GD.js";import"./isPlainObject-CjvgKE_V.js";import"./_baseUniq-DznNr2XC.js";const xe={title:"Components/CategoryMatchingChart",component:ce,parameters:{layout:"padded",docs:{description:{component:`CategoryMatchingChart displays category matching scores as horizontal bar charts.

## Color Coding
- **Green** (>0.7): High match score
- **Yellow** (0.3-0.7): Medium match score
- **Red** (<0.3): Low match score

## Features
- Shows at least top 3 categories
- Scores displayed as percentages
- Click to expand reasons
- Feature contributions breakdown
- Responsive layout

## Accessibility
- ARIA labels for chart elements
- Keyboard navigable
- Screen reader friendly

## Usage
Used in ReasoningPanel to show category matching reasoning.`}}},tags:["autodocs"],argTypes:{categories:{description:"Array of category matching data with scores and reasons"},onCategoryClick:{description:"Optional callback when a category is clicked",action:"category clicked"}}},l=[{category_name:"Elektronik",score:.85,reasons:["Hobi eşleşmesi: Teknoloji meraklısı","Yaş uygunluğu: Genç yetişkin","Bütçe uygunluğu: Orta-yüksek bütçe"],feature_contributions:{hobby:.9,age:.8,budget:.85}},{category_name:"Kitap",score:.65,reasons:["Hobi eşleşmesi: Okuma sevgisi","Yaş uygunluğu: Her yaş"],feature_contributions:{hobby:.7,age:.6}},{category_name:"Spor Malzemeleri",score:.45,reasons:["Hobi eşleşmesi: Aktif yaşam"],feature_contributions:{hobby:.5}},{category_name:"Ev Dekorasyonu",score:.25,reasons:["Düşük ilgi: Ev dekorasyonuna ilgi yok"],feature_contributions:{hobby:.2,occasion:.3}}],e={args:{categories:l}},o={args:{categories:[{category_name:"Teknoloji",score:.92,reasons:["Mükemmel hobi eşleşmesi","Yüksek bütçe uygunluğu"],feature_contributions:{hobby:.95,budget:.9}},{category_name:"Oyun",score:.88,reasons:["Güçlü hobi eşleşmesi","Yaş uygunluğu"],feature_contributions:{hobby:.9,age:.85}},{category_name:"Elektronik Aksesuarlar",score:.75,reasons:["İyi hobi eşleşmesi"],feature_contributions:{hobby:.8}}]}},r={args:{categories:[{category_name:"Bahçe",score:.28,reasons:["Düşük ilgi"],feature_contributions:{hobby:.2}},{category_name:"Bebek Ürünleri",score:.15,reasons:["İlgisiz kategori"],feature_contributions:{occasion:.1}},{category_name:"Otomotiv",score:.22,reasons:["Düşük eşleşme"],feature_contributions:{hobby:.25}}]}},n={args:{categories:l}},a={args:{categories:[{category_name:"Elektronik",score:.85,reasons:["Hobi eşleşmesi"],feature_contributions:{hobby:.9}},{category_name:"Kitap",score:.65,reasons:["Yaş uygunluğu"],feature_contributions:{age:.7}},{category_name:"Spor",score:.45,reasons:["Orta eşleşme"],feature_contributions:{hobby:.5}}]}},s={args:{categories:[{category_name:"Elektronik",score:.95,reasons:["Mükemmel eşleşme"],feature_contributions:{hobby:.95}},{category_name:"Kitap",score:.85,reasons:["Çok iyi eşleşme"],feature_contributions:{hobby:.85}},{category_name:"Oyun",score:.75,reasons:["İyi eşleşme"],feature_contributions:{hobby:.75}},{category_name:"Spor",score:.65,reasons:["Orta eşleşme"],feature_contributions:{hobby:.65}},{category_name:"Müzik",score:.55,reasons:["Orta-düşük eşleşme"],feature_contributions:{hobby:.55}},{category_name:"Moda",score:.45,reasons:["Düşük eşleşme"],feature_contributions:{hobby:.45}},{category_name:"Ev",score:.35,reasons:["Çok düşük eşleşme"],feature_contributions:{hobby:.35}},{category_name:"Bahçe",score:.25,reasons:["Minimal eşleşme"],feature_contributions:{hobby:.25}}]}},t={args:{categories:[]}},i={args:{categories:l,onCategoryClick:m=>{console.log("Category clicked:",m),alert(`Kategori tıklandı: ${m.category_name} (${(m.score*100).toFixed(0)}%)`)}}},c={args:{categories:[{category_name:"Elektronik",score:.85,reasons:["Hobi eşleşmesi"],feature_contributions:{}},{category_name:"Kitap",score:.65,reasons:["Yaş uygunluğu"],feature_contributions:{}},{category_name:"Spor",score:.45,reasons:["Orta eşleşme"],feature_contributions:{}}]}},u={args:{categories:[{category_name:"Elektronik",score:.85,reasons:["Tek neden: Hobi eşleşmesi"],feature_contributions:{hobby:.9}}]}},g={args:{categories:[{category_name:"Elektronik",score:.85,reasons:["Hobi eşleşmesi: Teknoloji meraklısı","Yaş uygunluğu: Genç yetişkin","Bütçe uygunluğu: Orta-yüksek bütçe","Occasion uygunluğu: Doğum günü hediyesi","Trend analizi: Popüler kategori"],feature_contributions:{hobby:.9,age:.8,budget:.85,occasion:.75,trend:.7}}]}};var y,b,d,p,k;e.parameters={...e.parameters,docs:{...(y=e.parameters)==null?void 0:y.docs,source:{originalSource:`{
  args: {
    categories: mockCategories
  }
}`,...(d=(b=e.parameters)==null?void 0:b.docs)==null?void 0:d.source},description:{story:`Default category matching with mixed scores
Shows green, yellow, and red bars based on score ranges`,...(k=(p=e.parameters)==null?void 0:p.docs)==null?void 0:k.description}}};var _,h,F,C,f;o.parameters={...o.parameters,docs:{...(_=o.parameters)==null?void 0:_.docs,source:{originalSource:`{
  args: {
    categories: [{
      category_name: 'Teknoloji',
      score: 0.92,
      reasons: ['Mükemmel hobi eşleşmesi', 'Yüksek bütçe uygunluğu'],
      feature_contributions: {
        hobby: 0.95,
        budget: 0.9
      }
    }, {
      category_name: 'Oyun',
      score: 0.88,
      reasons: ['Güçlü hobi eşleşmesi', 'Yaş uygunluğu'],
      feature_contributions: {
        hobby: 0.9,
        age: 0.85
      }
    }, {
      category_name: 'Elektronik Aksesuarlar',
      score: 0.75,
      reasons: ['İyi hobi eşleşmesi'],
      feature_contributions: {
        hobby: 0.8
      }
    }]
  }
}`,...(F=(h=o.parameters)==null?void 0:h.docs)==null?void 0:F.source},description:{story:`All categories with high scores (>0.7)
All bars should be green`,...(f=(C=o.parameters)==null?void 0:C.docs)==null?void 0:f.description}}};var x,S,E,M,O;r.parameters={...r.parameters,docs:{...(x=r.parameters)==null?void 0:x.docs,source:{originalSource:`{
  args: {
    categories: [{
      category_name: 'Bahçe',
      score: 0.28,
      reasons: ['Düşük ilgi'],
      feature_contributions: {
        hobby: 0.2
      }
    }, {
      category_name: 'Bebek Ürünleri',
      score: 0.15,
      reasons: ['İlgisiz kategori'],
      feature_contributions: {
        occasion: 0.1
      }
    }, {
      category_name: 'Otomotiv',
      score: 0.22,
      reasons: ['Düşük eşleşme'],
      feature_contributions: {
        hobby: 0.25
      }
    }]
  }
}`,...(E=(S=r.parameters)==null?void 0:S.docs)==null?void 0:E.source},description:{story:`All categories with low scores (<0.3)
All bars should be red`,...(O=(M=r.parameters)==null?void 0:M.docs)==null?void 0:O.description}}};var H,w,D,v,Y;n.parameters={...n.parameters,docs:{...(H=n.parameters)==null?void 0:H.docs,source:{originalSource:`{
  args: {
    categories: mockCategories
  }
}`,...(D=(w=n.parameters)==null?void 0:w.docs)==null?void 0:D.source},description:{story:"Mixed score ranges demonstrating all color codes",...(Y=(v=n.parameters)==null?void 0:v.docs)==null?void 0:Y.description}}};var A,K,T,B,z;a.parameters={...a.parameters,docs:{...(A=a.parameters)==null?void 0:A.docs,source:{originalSource:`{
  args: {
    categories: [{
      category_name: 'Elektronik',
      score: 0.85,
      reasons: ['Hobi eşleşmesi'],
      feature_contributions: {
        hobby: 0.9
      }
    }, {
      category_name: 'Kitap',
      score: 0.65,
      reasons: ['Yaş uygunluğu'],
      feature_contributions: {
        age: 0.7
      }
    }, {
      category_name: 'Spor',
      score: 0.45,
      reasons: ['Orta eşleşme'],
      feature_contributions: {
        hobby: 0.5
      }
    }]
  }
}`,...(T=(K=a.parameters)==null?void 0:K.docs)==null?void 0:T.source},description:{story:"Minimum of 3 categories (requirement validation)",...(z=(B=a.parameters)==null?void 0:B.docs)==null?void 0:z.description}}};var R,G,j,$,L;s.parameters={...s.parameters,docs:{...(R=s.parameters)==null?void 0:R.docs,source:{originalSource:`{
  args: {
    categories: [{
      category_name: 'Elektronik',
      score: 0.95,
      reasons: ['Mükemmel eşleşme'],
      feature_contributions: {
        hobby: 0.95
      }
    }, {
      category_name: 'Kitap',
      score: 0.85,
      reasons: ['Çok iyi eşleşme'],
      feature_contributions: {
        hobby: 0.85
      }
    }, {
      category_name: 'Oyun',
      score: 0.75,
      reasons: ['İyi eşleşme'],
      feature_contributions: {
        hobby: 0.75
      }
    }, {
      category_name: 'Spor',
      score: 0.65,
      reasons: ['Orta eşleşme'],
      feature_contributions: {
        hobby: 0.65
      }
    }, {
      category_name: 'Müzik',
      score: 0.55,
      reasons: ['Orta-düşük eşleşme'],
      feature_contributions: {
        hobby: 0.55
      }
    }, {
      category_name: 'Moda',
      score: 0.45,
      reasons: ['Düşük eşleşme'],
      feature_contributions: {
        hobby: 0.45
      }
    }, {
      category_name: 'Ev',
      score: 0.35,
      reasons: ['Çok düşük eşleşme'],
      feature_contributions: {
        hobby: 0.35
      }
    }, {
      category_name: 'Bahçe',
      score: 0.25,
      reasons: ['Minimal eşleşme'],
      feature_contributions: {
        hobby: 0.25
      }
    }]
  }
}`,...(j=(G=s.parameters)==null?void 0:G.docs)==null?void 0:j.source},description:{story:"Many categories (8+) to test scrolling behavior",...(L=($=s.parameters)==null?void 0:$.docs)==null?void 0:L.description}}};var P,I,N,U,W;t.parameters={...t.parameters,docs:{...(P=t.parameters)==null?void 0:P.docs,source:{originalSource:`{
  args: {
    categories: []
  }
}`,...(N=(I=t.parameters)==null?void 0:I.docs)==null?void 0:N.source},description:{story:"Empty state - no categories available",...(W=(U=t.parameters)==null?void 0:U.docs)==null?void 0:W.description}}};var q,J,Q,V,X;i.parameters={...i.parameters,docs:{...(q=i.parameters)==null?void 0:q.docs,source:{originalSource:`{
  args: {
    categories: mockCategories,
    onCategoryClick: category => {
      console.log('Category clicked:', category);
      alert(\`Kategori tıklandı: \${category.category_name} (\${(category.score * 100).toFixed(0)}%)\`);
    }
  }
}`,...(Q=(J=i.parameters)==null?void 0:J.docs)==null?void 0:Q.source},description:{story:`Interactive category click handler
Click on a category to see its details`,...(X=(V=i.parameters)==null?void 0:V.docs)==null?void 0:X.description}}};var Z,ee,oe;c.parameters={...c.parameters,docs:{...(Z=c.parameters)==null?void 0:Z.docs,source:{originalSource:`{
  args: {
    categories: [{
      category_name: 'Elektronik',
      score: 0.85,
      reasons: ['Hobi eşleşmesi'],
      feature_contributions: {}
    }, {
      category_name: 'Kitap',
      score: 0.65,
      reasons: ['Yaş uygunluğu'],
      feature_contributions: {}
    }, {
      category_name: 'Spor',
      score: 0.45,
      reasons: ['Orta eşleşme'],
      feature_contributions: {}
    }]
  }
}`,...(oe=(ee=c.parameters)==null?void 0:ee.docs)==null?void 0:oe.source}}};var re,ne,ae;u.parameters={...u.parameters,docs:{...(re=u.parameters)==null?void 0:re.docs,source:{originalSource:`{
  args: {
    categories: [{
      category_name: 'Elektronik',
      score: 0.85,
      reasons: ['Tek neden: Hobi eşleşmesi'],
      feature_contributions: {
        hobby: 0.9
      }
    }]
  }
}`,...(ae=(ne=u.parameters)==null?void 0:ne.docs)==null?void 0:ae.source}}};var se,te,ie;g.parameters={...g.parameters,docs:{...(se=g.parameters)==null?void 0:se.docs,source:{originalSource:`{
  args: {
    categories: [{
      category_name: 'Elektronik',
      score: 0.85,
      reasons: ['Hobi eşleşmesi: Teknoloji meraklısı', 'Yaş uygunluğu: Genç yetişkin', 'Bütçe uygunluğu: Orta-yüksek bütçe', 'Occasion uygunluğu: Doğum günü hediyesi', 'Trend analizi: Popüler kategori'],
      feature_contributions: {
        hobby: 0.9,
        age: 0.8,
        budget: 0.85,
        occasion: 0.75,
        trend: 0.7
      }
    }]
  }
}`,...(ie=(te=g.parameters)==null?void 0:te.docs)==null?void 0:ie.source}}};const Se=["Default","HighScoresOnly","LowScoresOnly","MixedScores","MinimumCategories","ManyCategories","EmptyState","WithClickHandler","NoFeatureContributions","SingleReason","MultipleReasons"];export{e as Default,t as EmptyState,o as HighScoresOnly,r as LowScoresOnly,s as ManyCategories,a as MinimumCategories,n as MixedScores,g as MultipleReasons,c as NoFeatureContributions,u as SingleReason,i as WithClickHandler,Se as __namedExportsOrder,xe as default};
