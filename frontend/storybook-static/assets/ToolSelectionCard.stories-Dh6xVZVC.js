import{T as O}from"./ToolSelectionCard-Bl_vwriK.js";import"./jsx-runtime-Bywvkw1S.js";import"./index-CleY8y_P.js";import"./_commonjsHelpers-Cpj98o6Y.js";import"./cn-CytzSlOG.js";import"./index-DBU18h-i.js";import"./index-BAUB8I8r.js";import"./index-CuYJjLYj.js";const $={title:"Components/ToolSelectionCard",component:O,parameters:{layout:"centered",docs:{description:{component:`ToolSelectionCard displays which tools were selected by the model and why.

## Visual Indicators
- **Green + Checkmark**: Selected tool
- **Gray**: Unselected tool
- **Yellow tooltip**: Low confidence warning

## Features
- Shows selection status, confidence score, and priority
- Hover tooltips with selection reasons and factors
- Low confidence warnings
- Responsive layout

## Accessibility
- ARIA labels for screen readers
- Keyboard navigable tooltips
- Semantic HTML structure

## Usage
Used in the ReasoningPanel to show tool selection reasoning.`}}},tags:["autodocs"],argTypes:{toolSelection:{description:"Array of tool selection reasoning data"}}},a=[{name:"review_analysis",selected:!0,score:.85,reason:"Yüksek rating eşleşmesi ve pozitif yorumlar",confidence:.9,priority:1,factors:{rating_match:.9,review_sentiment:.85,review_count:.8}},{name:"trend_analysis",selected:!0,score:.72,reason:"Popüler trend kategorisinde",confidence:.75,priority:2,factors:{trend_score:.8,popularity:.7}},{name:"inventory_check",selected:!1,score:.45,reason:"Stok durumu belirsiz",confidence:.4,priority:3,factors:{availability:.5,stock_level:.4}},{name:"price_comparison",selected:!0,score:.88,reason:"Bütçe ile mükemmel uyum",confidence:.92,priority:4,factors:{budget_match:.95,price_competitiveness:.85}}],e={args:{toolSelection:a}},o={args:{toolSelection:a.map(i=>({...i,selected:!0,confidence:.85}))}},n={args:{toolSelection:a.map(i=>({...i,selected:!1,confidence:.6}))}},r={args:{toolSelection:[{name:"review_analysis",selected:!0,score:.35,reason:"Sınırlı yorum verisi",confidence:.3,priority:1,factors:{review_count:.2}},{name:"trend_analysis",selected:!1,score:.25,reason:"Trend verisi yetersiz",confidence:.2,priority:2}]}},s={args:{toolSelection:[{name:"review_analysis",selected:!0,score:.75,reason:"Genel değerlendirme olumlu",confidence:.8,priority:1},{name:"inventory_check",selected:!1,score:.5,reason:"Stok bilgisi alınamadı",confidence:.5,priority:2}]}},t={args:{toolSelection:[]}},c={args:{toolSelection:[{name:"price_comparison",selected:!0,score:.95,reason:"Mükemmel fiyat uyumu",confidence:.98,priority:1,factors:{budget_match:1,price_competitiveness:.9}}]}};var l,d,p,m,u;e.parameters={...e.parameters,docs:{...(l=e.parameters)==null?void 0:l.docs,source:{originalSource:`{
  args: {
    toolSelection: sampleToolSelection
  }
}`,...(p=(d=e.parameters)==null?void 0:d.docs)==null?void 0:p.source},description:{story:"Default tool selection with mixed selected/unselected tools",...(u=(m=e.parameters)==null?void 0:m.docs)==null?void 0:u.description}}};var y,f,g,S,w;o.parameters={...o.parameters,docs:{...(y=o.parameters)==null?void 0:y.docs,source:{originalSource:`{
  args: {
    toolSelection: sampleToolSelection.map(tool => ({
      ...tool,
      selected: true,
      confidence: 0.85
    }))
  }
}`,...(g=(f=o.parameters)==null?void 0:f.docs)==null?void 0:g.source},description:{story:"All tools selected with high confidence",...(w=(S=o.parameters)==null?void 0:S.docs)==null?void 0:w.description}}};var _,h,v,T,k;n.parameters={...n.parameters,docs:{...(_=n.parameters)==null?void 0:_.docs,source:{originalSource:`{
  args: {
    toolSelection: sampleToolSelection.map(tool => ({
      ...tool,
      selected: false,
      confidence: 0.6
    }))
  }
}`,...(v=(h=n.parameters)==null?void 0:h.docs)==null?void 0:v.source},description:{story:"No tools selected (all gray)",...(k=(T=n.parameters)==null?void 0:T.docs)==null?void 0:k.description}}};var b,A,C,F,L;r.parameters={...r.parameters,docs:{...(b=r.parameters)==null?void 0:b.docs,source:{originalSource:`{
  args: {
    toolSelection: [{
      name: 'review_analysis',
      selected: true,
      score: 0.35,
      reason: 'Sınırlı yorum verisi',
      confidence: 0.3,
      priority: 1,
      factors: {
        review_count: 0.2
      }
    }, {
      name: 'trend_analysis',
      selected: false,
      score: 0.25,
      reason: 'Trend verisi yetersiz',
      confidence: 0.2,
      priority: 2
    }]
  }
}`,...(C=(A=r.parameters)==null?void 0:A.docs)==null?void 0:C.source},description:{story:`Tools with low confidence scores (<0.5)
Shows warning tooltips for low confidence`,...(L=(F=r.parameters)==null?void 0:F.docs)==null?void 0:L.description}}};var U,x,z,E,G;s.parameters={...s.parameters,docs:{...(U=s.parameters)==null?void 0:U.docs,source:{originalSource:`{
  args: {
    toolSelection: [{
      name: 'review_analysis',
      selected: true,
      score: 0.75,
      reason: 'Genel değerlendirme olumlu',
      confidence: 0.8,
      priority: 1
    }, {
      name: 'inventory_check',
      selected: false,
      score: 0.5,
      reason: 'Stok bilgisi alınamadı',
      confidence: 0.5,
      priority: 2
    }]
  }
}`,...(z=(x=s.parameters)==null?void 0:x.docs)==null?void 0:z.source},description:{story:`Tools without factor details
Tests graceful handling of missing factors`,...(G=(E=s.parameters)==null?void 0:E.docs)==null?void 0:G.description}}};var D,M,N,R,H;t.parameters={...t.parameters,docs:{...(D=t.parameters)==null?void 0:D.docs,source:{originalSource:`{
  args: {
    toolSelection: []
  }
}`,...(N=(M=t.parameters)==null?void 0:M.docs)==null?void 0:N.source},description:{story:"Empty state - no tools available",...(H=(R=t.parameters)==null?void 0:R.docs)==null?void 0:H.description}}};var I,P,Y,B,K;c.parameters={...c.parameters,docs:{...(I=c.parameters)==null?void 0:I.docs,source:{originalSource:`{
  args: {
    toolSelection: [{
      name: 'price_comparison',
      selected: true,
      score: 0.95,
      reason: 'Mükemmel fiyat uyumu',
      confidence: 0.98,
      priority: 1,
      factors: {
        budget_match: 1.0,
        price_competitiveness: 0.9
      }
    }]
  }
}`,...(Y=(P=c.parameters)==null?void 0:P.docs)==null?void 0:Y.source},description:{story:"Single tool with very high confidence",...(K=(B=c.parameters)==null?void 0:B.docs)==null?void 0:K.description}}};const ee=["Default","AllSelected","AllUnselected","LowConfidence","NoFactors","Empty","SingleTool"];export{o as AllSelected,n as AllUnselected,e as Default,t as Empty,r as LowConfidence,s as NoFactors,c as SingleTool,ee as __namedExportsOrder,$ as default};
