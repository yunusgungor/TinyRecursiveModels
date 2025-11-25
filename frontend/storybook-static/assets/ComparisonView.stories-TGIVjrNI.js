import{j as e}from"./jsx-runtime-Bywvkw1S.js";import{R as Se}from"./index-CleY8y_P.js";import{c as Ee}from"./cn-CytzSlOG.js";import{A as Ke}from"./AttentionWeightsChart-B5l3Q2E_.js";import{C as E}from"./ConfidenceIndicator-CRl75YCL.js";import{L as Te}from"./LazyImage-BPW1JQg4.js";import"./_commonjsHelpers-Cpj98o6Y.js";import"./useMediaQuery-Dj87e1Qj.js";import"./throttle-BoqV9_75.js";import"./mapValues-yqX3NJIo.js";import"./tiny-invariant-CopsF_GD.js";import"./isPlainObject-CjvgKE_V.js";import"./_baseUniq-DznNr2XC.js";const R=({recommendations:r,onExit:j,className:k})=>{const[_,N]=Se.useState("bar"),w=a=>new Intl.NumberFormat("tr-TR",{style:"currency",currency:"TRY",minimumFractionDigits:2,maximumFractionDigits:2}).format(a),o=(()=>{if(r.length===0)return[];const a=new Set;return r.forEach(s=>{var i;(i=s.reasoning_trace)==null||i.category_matching.forEach(c=>{a.add(c.category_name)})}),Array.from(a).map(s=>{const i={category:s};return r.forEach((c,Re)=>{var S;const C=(S=c.reasoning_trace)==null?void 0:S.category_matching.find(Ce=>Ce.category_name===s);i[`gift${Re+1}`]=C?C.score*100:0}),i})})();return e.jsxs("div",{className:Ee("comparison-view bg-white dark:bg-gray-900 min-h-screen p-6",k),role:"region","aria-label":"Hediye karşılaştırma görünümü",children:[e.jsxs("div",{className:"flex items-center justify-between mb-6",children:[e.jsx("h2",{className:"text-2xl font-bold text-gray-900 dark:text-white",children:"Hediye Karşılaştırma"}),e.jsx("button",{onClick:j,className:"px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500","aria-label":"Karşılaştırma modundan çık",children:"Karşılaştırmayı Kapat"})]}),e.jsx("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8",children:r.map((a,s)=>e.jsxs("div",{className:"bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-4",role:"article","aria-label":`Hediye ${s+1}: ${a.gift.name}`,children:[e.jsxs("div",{className:"relative aspect-square w-full overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-700 mb-4",children:[e.jsx(Te,{src:a.gift.image_url||"",alt:a.gift.name,className:"w-full h-full object-cover",placeholderClassName:"aspect-square"}),e.jsx("div",{className:"absolute top-2 right-2",children:e.jsx(E,{confidence:a.confidence})})]}),e.jsxs("div",{className:"space-y-2",children:[e.jsx("h3",{className:"text-lg font-semibold text-gray-900 dark:text-white line-clamp-2",children:a.gift.name}),e.jsx("p",{className:"text-xl font-bold text-gray-900 dark:text-white",children:w(a.gift.price)}),e.jsx("span",{className:"inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-medium px-2.5 py-0.5 rounded",children:a.gift.category})]}),e.jsx("div",{className:"mt-4 space-y-1",children:a.reasoning.slice(0,2).map((i,c)=>e.jsxs("p",{className:"text-sm text-gray-700 dark:text-gray-300",children:["• ",i]},c))})]},a.gift.id))}),e.jsxs("div",{className:"space-y-8",children:[o.length>0&&e.jsxs("div",{className:"bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-6",children:[e.jsx("h3",{className:"text-xl font-semibold text-gray-900 dark:text-white mb-4",children:"Kategori Skorları Karşılaştırması"}),e.jsx("div",{className:"overflow-x-auto",children:e.jsx(Me,{data:o,giftNames:r.map(a=>a.gift.name)})})]}),r.every(a=>{var s;return(s=a.reasoning_trace)==null?void 0:s.attention_weights})&&e.jsxs("div",{className:"bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-6",children:[e.jsx("h3",{className:"text-xl font-semibold text-gray-900 dark:text-white mb-4",children:"Attention Weights Karşılaştırması"}),e.jsx("div",{className:"grid grid-cols-1 lg:grid-cols-2 gap-6",children:r.map((a,s)=>{var i;return e.jsxs("div",{children:[e.jsx("h4",{className:"text-md font-medium text-gray-800 dark:text-gray-200 mb-2",children:a.gift.name}),((i=a.reasoning_trace)==null?void 0:i.attention_weights)&&e.jsx(Ke,{attentionWeights:a.reasoning_trace.attention_weights,chartType:_,onChartTypeChange:c=>N(c)})]},a.gift.id)})})]}),e.jsxs("div",{className:"bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-6",children:[e.jsx("h3",{className:"text-xl font-semibold text-gray-900 dark:text-white mb-4",children:"Güven Skoru Karşılaştırması"}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("table",{className:"min-w-full divide-y divide-gray-200 dark:divide-gray-700",children:[e.jsx("thead",{className:"bg-gray-100 dark:bg-gray-700",children:e.jsxs("tr",{children:[e.jsx("th",{className:"px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider",children:"Hediye"}),e.jsx("th",{className:"px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider",children:"Güven Skoru"}),e.jsx("th",{className:"px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider",children:"Fiyat"})]})}),e.jsx("tbody",{className:"bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700",children:r.map(a=>e.jsxs("tr",{children:[e.jsx("td",{className:"px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white",children:a.gift.name}),e.jsx("td",{className:"px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300",children:e.jsxs("div",{className:"flex items-center gap-2",children:[e.jsx(E,{confidence:a.confidence}),e.jsxs("span",{children:[(a.confidence*100).toFixed(0),"%"]})]})}),e.jsx("td",{className:"px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300",children:w(a.gift.price)})]},a.gift.id))})]})})]})]})]})},Me=({data:r,giftNames:j})=>{const k=["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"];return e.jsx("div",{className:"w-full",style:{minHeight:"400px"},children:r.map((_,N)=>e.jsxs("div",{className:"mb-4",children:[e.jsx("div",{className:"flex items-center justify-between mb-2",children:e.jsx("span",{className:"text-sm font-medium text-gray-700 dark:text-gray-300",children:_.category})}),e.jsx("div",{className:"space-y-2",children:j.map((w,v)=>{const o=_[`gift${v+1}`]||0,a=k[v%k.length];return e.jsxs("div",{className:"flex items-center gap-2",children:[e.jsx("div",{className:"w-32 text-xs text-gray-600 dark:text-gray-400 truncate",children:w}),e.jsx("div",{className:"flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative",children:e.jsx("div",{className:"h-6 rounded-full flex items-center justify-end pr-2 text-xs font-medium text-white transition-all duration-300",style:{width:`${o}%`,backgroundColor:a},children:o>10&&`${o.toFixed(0)}%`})}),o<=10&&e.jsxs("span",{className:"text-xs text-gray-600 dark:text-gray-400 w-12",children:[o.toFixed(0),"%"]})]},v)})})]},N))})};try{R.displayName="ComparisonView",R.__docgenInfo={description:"Displays side-by-side comparison of selected gift recommendations",displayName:"ComparisonView",props:{recommendations:{defaultValue:null,description:"",name:"recommendations",required:!0,type:{name:"EnhancedGiftRecommendation[]"}},onExit:{defaultValue:null,description:"",name:"onExit",required:!0,type:{name:"() => void"}},className:{defaultValue:null,description:"",name:"className",required:!1,type:{name:"string"}}}}}catch{}const We={title:"Components/ComparisonView",component:R,parameters:{layout:"fullscreen",docs:{description:{component:`ComparisonView displays a side-by-side comparison of selected gift recommendations.

## Features
- Side-by-side gift card display
- Category scores comparison chart
- Attention weights comparison
- Confidence score comparison table
- Responsive layout (stacks on mobile)

## Accessibility
- ARIA labels for screen readers
- Keyboard navigable
- Semantic HTML structure
- Focus management

## Usage
Used when users select multiple gifts for comparison. Typically accessed
from the recommendations page when 2 or more gifts are selected.`}}},tags:["autodocs"],argTypes:{onExit:{action:"exit comparison",description:"Callback when user exits comparison mode"}}},t={gift:{id:"gift-1",name:"Premium Kahve Makinesi",price:1500,category:"Ev & Yaşam",image_url:"https://via.placeholder.com/400x400/3b82f6/ffffff?text=Kahve+Makinesi",rating:4.5,availability:!0},reasoning:["Kullanıcının kahve hobisi ile mükemmel eşleşme","Bütçe aralığına uygun fiyat","Yüksek rating ve pozitif yorumlar"],confidence:.92,reasoning_trace:{tool_selection:[{name:"review_analysis",selected:!0,score:.85,reason:"Yüksek rating",confidence:.9,priority:1}],category_matching:[{category_name:"Ev & Yaşam",score:.88,reasons:["Kahve hobisi ile uyumlu"],feature_contributions:{hobby_match:.9,age_appropriateness:.85}},{category_name:"Mutfak",score:.82,reasons:["Yemek pişirme hobisi ile uyumlu"],feature_contributions:{hobby_match:.85}},{category_name:"Teknoloji",score:.65,reasons:["Modern cihaz"],feature_contributions:{hobby_match:.7}}],attention_weights:{user_features:{hobbies:.4,budget:.3,age:.2,occasion:.1},gift_features:{category:.5,price:.3,rating:.2}},thinking_steps:[]}},n={gift:{id:"gift-2",name:"Profesyonel Espresso Makinesi",price:2200,category:"Ev & Yaşam",image_url:"https://via.placeholder.com/400x400/10b981/ffffff?text=Espresso+Makinesi",rating:4.8,availability:!0},reasoning:["Üst segment kahve deneyimi","Profesyonel özellikler","Mükemmel kullanıcı değerlendirmeleri"],confidence:.85,reasoning_trace:{tool_selection:[{name:"review_analysis",selected:!0,score:.92,reason:"Çok yüksek rating",confidence:.95,priority:1}],category_matching:[{category_name:"Ev & Yaşam",score:.85,reasons:["Kahve hobisi ile uyumlu"],feature_contributions:{hobby_match:.88,age_appropriateness:.82}},{category_name:"Mutfak",score:.78,reasons:["Mutfak ekipmanı"],feature_contributions:{hobby_match:.8}},{category_name:"Teknoloji",score:.72,reasons:["Gelişmiş teknoloji"],feature_contributions:{hobby_match:.75}}],attention_weights:{user_features:{hobbies:.35,budget:.25,age:.25,occasion:.15},gift_features:{category:.45,price:.35,rating:.2}},thinking_steps:[]}},b={gift:{id:"gift-3",name:"Kompakt Kahve Makinesi",price:899,category:"Ev & Yaşam",image_url:"https://via.placeholder.com/400x400/f59e0b/ffffff?text=Kompakt+Kahve",rating:4.2,availability:!0},reasoning:["Ekonomik seçenek","Kompakt tasarım","İyi değerlendirmeler"],confidence:.68,reasoning_trace:{tool_selection:[{name:"review_analysis",selected:!0,score:.75,reason:"İyi rating",confidence:.8,priority:1}],category_matching:[{category_name:"Ev & Yaşam",score:.75,reasons:["Kahve hobisi ile uyumlu"],feature_contributions:{hobby_match:.78,age_appropriateness:.72}},{category_name:"Mutfak",score:.68,reasons:["Mutfak ekipmanı"],feature_contributions:{hobby_match:.7}},{category_name:"Teknoloji",score:.55,reasons:["Temel teknoloji"],feature_contributions:{hobby_match:.6}}],attention_weights:{user_features:{hobbies:.3,budget:.4,age:.2,occasion:.1},gift_features:{category:.4,price:.4,rating:.2}},thinking_steps:[]}},m={args:{recommendations:[t,n]}},d={args:{recommendations:[t,n,b]}},l={args:{recommendations:[{...t,confidence:.95},{...n,confidence:.92}]}},g={args:{recommendations:[{...t,confidence:.92},{...n,confidence:.65},{...b,confidence:.35}]}},p={args:{recommendations:[{...t,gift:{...t.gift,price:500}},{...n,gift:{...n.gift,price:1500}},{...b,gift:{...b.gift,price:3e3}}]}},f={args:{recommendations:[{...t,confidence:.85,reasoning_trace:{...t.reasoning_trace,category_matching:t.reasoning_trace.category_matching.map(r=>({...r,score:.8}))}},{...n,confidence:.83,reasoning_trace:{...n.reasoning_trace,category_matching:n.reasoning_trace.category_matching.map(r=>({...r,score:.78}))}}]}},h={args:{recommendations:[t,n]},parameters:{viewport:{defaultViewport:"mobile1"}}},u={args:{recommendations:[t,n,b]},parameters:{viewport:{defaultViewport:"tablet"}}},y={args:{recommendations:[]}},x={args:{recommendations:[t]}};var K,T,M,V,G;m.parameters={...m.parameters,docs:{...(K=m.parameters)==null?void 0:K.docs,source:{originalSource:`{
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2]
  }
}`,...(M=(T=m.parameters)==null?void 0:T.docs)==null?void 0:M.source},description:{story:"Default comparison view with two gifts",...(G=(V=m.parameters)==null?void 0:V.docs)==null?void 0:G.description}}};var Y,D,F,A,H;d.parameters={...d.parameters,docs:{...(Y=d.parameters)==null?void 0:Y.docs,source:{originalSource:`{
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2, mockRecommendation3]
  }
}`,...(F=(D=d.parameters)==null?void 0:D.docs)==null?void 0:F.source},description:{story:"Comparison view with three gifts",...(H=(A=d.parameters)==null?void 0:A.docs)==null?void 0:H.description}}};var P,$,q,z,I;l.parameters={...l.parameters,docs:{...(P=l.parameters)==null?void 0:P.docs,source:{originalSource:`{
  args: {
    recommendations: [{
      ...mockRecommendation1,
      confidence: 0.95
    }, {
      ...mockRecommendation2,
      confidence: 0.92
    }]
  }
}`,...(q=($=l.parameters)==null?void 0:$.docs)==null?void 0:q.source},description:{story:"Comparison with high confidence gifts",...(I=(z=l.parameters)==null?void 0:z.docs)==null?void 0:I.description}}};var L,W,U,B,O;g.parameters={...g.parameters,docs:{...(L=g.parameters)==null?void 0:L.docs,source:{originalSource:`{
  args: {
    recommendations: [{
      ...mockRecommendation1,
      confidence: 0.92
    }, {
      ...mockRecommendation2,
      confidence: 0.65
    }, {
      ...mockRecommendation3,
      confidence: 0.35
    }]
  }
}`,...(U=(W=g.parameters)==null?void 0:W.docs)==null?void 0:U.source},description:{story:"Comparison with mixed confidence levels",...(O=(B=g.parameters)==null?void 0:B.docs)==null?void 0:O.description}}};var J,Q,X,Z,ee;p.parameters={...p.parameters,docs:{...(J=p.parameters)==null?void 0:J.docs,source:{originalSource:`{
  args: {
    recommendations: [{
      ...mockRecommendation1,
      gift: {
        ...mockRecommendation1.gift,
        price: 500
      }
    }, {
      ...mockRecommendation2,
      gift: {
        ...mockRecommendation2.gift,
        price: 1500
      }
    }, {
      ...mockRecommendation3,
      gift: {
        ...mockRecommendation3.gift,
        price: 3000
      }
    }]
  }
}`,...(X=(Q=p.parameters)==null?void 0:Q.docs)==null?void 0:X.source},description:{story:"Comparison with different price ranges",...(ee=(Z=p.parameters)==null?void 0:Z.docs)==null?void 0:ee.description}}};var ae,re,te,ne,se;f.parameters={...f.parameters,docs:{...(ae=f.parameters)==null?void 0:ae.docs,source:{originalSource:`{
  args: {
    recommendations: [{
      ...mockRecommendation1,
      confidence: 0.85,
      reasoning_trace: {
        ...mockRecommendation1.reasoning_trace!,
        category_matching: mockRecommendation1.reasoning_trace!.category_matching.map(cat => ({
          ...cat,
          score: 0.8
        }))
      }
    }, {
      ...mockRecommendation2,
      confidence: 0.83,
      reasoning_trace: {
        ...mockRecommendation2.reasoning_trace!,
        category_matching: mockRecommendation2.reasoning_trace!.category_matching.map(cat => ({
          ...cat,
          score: 0.78
        }))
      }
    }]
  }
}`,...(te=(re=f.parameters)==null?void 0:re.docs)==null?void 0:te.source},description:{story:"Comparison with similar category scores",...(se=(ne=f.parameters)==null?void 0:ne.docs)==null?void 0:se.description}}};var ie,oe,ce,me,de;h.parameters={...h.parameters,docs:{...(ie=h.parameters)==null?void 0:ie.docs,source:{originalSource:`{
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2]
  },
  parameters: {
    viewport: {
      defaultViewport: 'mobile1'
    }
  }
}`,...(ce=(oe=h.parameters)==null?void 0:oe.docs)==null?void 0:ce.source},description:{story:"Mobile viewport comparison",...(de=(me=h.parameters)==null?void 0:me.docs)==null?void 0:de.description}}};var le,ge,pe,fe,he;u.parameters={...u.parameters,docs:{...(le=u.parameters)==null?void 0:le.docs,source:{originalSource:`{
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2, mockRecommendation3]
  },
  parameters: {
    viewport: {
      defaultViewport: 'tablet'
    }
  }
}`,...(pe=(ge=u.parameters)==null?void 0:ge.docs)==null?void 0:pe.source},description:{story:"Tablet viewport comparison",...(he=(fe=u.parameters)==null?void 0:fe.docs)==null?void 0:he.description}}};var ue,ye,xe,be,ke;y.parameters={...y.parameters,docs:{...(ue=y.parameters)==null?void 0:ue.docs,source:{originalSource:`{
  args: {
    recommendations: []
  }
}`,...(xe=(ye=y.parameters)==null?void 0:ye.docs)==null?void 0:xe.source},description:{story:"Empty state (should not happen in practice)",...(ke=(be=y.parameters)==null?void 0:be.docs)==null?void 0:ke.description}}};var _e,we,ve,je,Ne;x.parameters={...x.parameters,docs:{...(_e=x.parameters)==null?void 0:_e.docs,source:{originalSource:`{
  args: {
    recommendations: [mockRecommendation1]
  }
}`,...(ve=(we=x.parameters)==null?void 0:we.docs)==null?void 0:ve.source},description:{story:"Single gift (edge case)",...(Ne=(je=x.parameters)==null?void 0:je.docs)==null?void 0:Ne.description}}};const Ue=["TwoGifts","ThreeGifts","HighConfidenceComparison","MixedConfidence","DifferentPriceRanges","SimilarScores","MobileView","TabletView","EmptyState","SingleGift"];export{p as DifferentPriceRanges,y as EmptyState,l as HighConfidenceComparison,g as MixedConfidence,h as MobileView,f as SimilarScores,x as SingleGift,u as TabletView,d as ThreeGifts,m as TwoGifts,Ue as __namedExportsOrder,We as default};
