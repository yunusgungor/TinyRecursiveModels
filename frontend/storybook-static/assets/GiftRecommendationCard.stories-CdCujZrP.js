import{j as e}from"./jsx-runtime-Bywvkw1S.js";import{r as ve}from"./index-CleY8y_P.js";import{c as Re}from"./cn-CytzSlOG.js";import{C as je}from"./ConfidenceIndicator-CRl75YCL.js";import{L as Ne}from"./LazyImage-BPW1JQg4.js";import"./_commonjsHelpers-Cpj98o6Y.js";const _=({recommendation:C,toolResults:r,onShowDetails:S,isSelected:T,onSelect:F,className:de})=>{const{gift:t,reasoning:m,confidence:ue}=C,[l,ge]=ve.useState(!1),pe=m.join(" ").length>200,fe=s=>new Intl.NumberFormat("tr-TR",{style:"currency",currency:"TRY",minimumFractionDigits:2,maximumFractionDigits:2}).format(s),xe=s=>{const i=[{pattern:/hobi|ilgi alanı|ilgi/gi,className:"text-purple-600 dark:text-purple-400 font-medium"},{pattern:/bütçe|fiyat|uygun/gi,className:"text-green-600 dark:text-green-400 font-medium"},{pattern:/yaş|yaşa uygun/gi,className:"text-blue-600 dark:text-blue-400 font-medium"}];let N=[s];return i.forEach(({pattern:ke,className:be})=>{N=N.flatMap(o=>{if(typeof o!="string")return o;const z=[...o.matchAll(ke)];if(z.length===0)return o;const d=[];let c=0;return z.forEach((w,ye)=>{const u=w.index;u>c&&d.push(o.substring(c,u)),d.push(e.jsx("span",{className:be,children:w[0]},`${u}-${ye}`)),c=u+w[0].length}),c<o.length&&d.push(o.substring(c)),d})}),e.jsx(e.Fragment,{children:N})},he=()=>{var s,i;return r?e.jsxs("div",{className:"flex items-center gap-2 mt-3",role:"list","aria-label":"Tool insights",children:[r.review_analysis&&e.jsxs("div",{className:"flex items-center gap-1 text-yellow-500",role:"listitem","aria-label":`Rating: ${r.review_analysis.average_rating}/5.0`,title:`Rating: ${r.review_analysis.average_rating}/5.0`,children:[e.jsx("svg",{className:"w-5 h-5",fill:"currentColor",viewBox:"0 0 20 20","aria-hidden":"true",children:e.jsx("path",{d:"M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"})}),e.jsx("span",{className:"text-sm font-medium",children:r.review_analysis.average_rating.toFixed(1)})]}),((s=r.trend_analysis)==null?void 0:s.trending)&&e.jsxs("div",{className:"flex items-center gap-1 text-green-500",role:"listitem","aria-label":"Trending",title:"Trending",children:[e.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24","aria-hidden":"true",children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"})}),e.jsx("span",{className:"text-sm font-medium",children:"Trend"})]}),((i=r.inventory_check)==null?void 0:i.available)&&e.jsxs("div",{className:"flex items-center gap-1 text-blue-500",role:"listitem","aria-label":"In Stock",title:"In Stock",children:[e.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24","aria-hidden":"true",children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"})}),e.jsx("span",{className:"text-sm font-medium",children:"Stokta"})]})]}):null};return e.jsxs("div",{className:Re("bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col",T&&"ring-2 ring-blue-500",de),role:"article","aria-label":`Gift recommendation: ${t.name}`,children:[e.jsxs("div",{className:"relative aspect-square w-full overflow-hidden bg-gray-100 dark:bg-gray-700",children:[e.jsx(Ne,{src:t.image_url||"",alt:t.name,className:"w-full h-full object-cover hover:scale-105 transition-transform duration-300",placeholderClassName:"aspect-square"}),e.jsx("div",{className:"absolute top-2 right-2",children:e.jsx(je,{confidence:ue})})]}),e.jsxs("div",{className:"p-4 flex flex-col flex-grow",children:[e.jsx("div",{className:"mb-2",children:e.jsx("span",{className:"inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-medium px-2.5 py-0.5 rounded",children:t.category})}),e.jsx("h3",{className:"text-lg font-semibold text-gray-900 dark:text-white mb-2 line-clamp-2 min-h-[3.5rem]",children:t.name}),e.jsx("div",{className:"mb-3",children:e.jsx("p",{className:"text-2xl font-bold text-gray-900 dark:text-white",children:fe(t.price)})}),m&&m.length>0&&e.jsxs("div",{className:"mb-3 flex-grow",role:"region","aria-label":"Reasoning information",children:[e.jsx("div",{className:"text-sm text-gray-700 dark:text-gray-300 space-y-1",children:m.slice(0,l?void 0:2).map((s,i)=>e.jsx("p",{className:"leading-relaxed",children:xe(s)},i))}),pe&&e.jsx("button",{onClick:()=>ge(!l),className:"mt-2 text-sm text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded","aria-expanded":l,"aria-label":l?"Daha az göster":"Daha fazla göster",children:l?"Daha az göster":"Daha fazla göster"})]}),he(),e.jsxs("div",{className:"flex flex-col gap-2 mt-4",children:[S&&e.jsx("button",{onClick:S,className:"w-full bg-blue-600 text-white py-2 px-4 rounded-md font-medium hover:bg-blue-700 active:bg-blue-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2","aria-label":"Detaylı analiz göster",children:"Detaylı Analiz Göster"}),F&&e.jsxs("label",{className:"flex items-center gap-2 cursor-pointer",children:[e.jsx("input",{type:"checkbox",checked:T,onChange:F,className:"w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2","aria-label":"Karşılaştırma için seç"}),e.jsx("span",{className:"text-sm text-gray-700 dark:text-gray-300",children:"Karşılaştırma için seç"})]})]})]})]})};try{_.displayName="GiftRecommendationCard",_.__docgenInfo={description:"Displays a gift recommendation card with reasoning information",displayName:"GiftRecommendationCard",props:{recommendation:{defaultValue:null,description:"",name:"recommendation",required:!0,type:{name:"EnhancedGiftRecommendation"}},toolResults:{defaultValue:null,description:"",name:"toolResults",required:!1,type:{name:"Record<string, any>"}},onShowDetails:{defaultValue:null,description:"",name:"onShowDetails",required:!1,type:{name:"(() => void)"}},isSelected:{defaultValue:null,description:"",name:"isSelected",required:!1,type:{name:"boolean"}},onSelect:{defaultValue:null,description:"",name:"onSelect",required:!1,type:{name:"(() => void)"}},className:{defaultValue:null,description:"",name:"className",required:!1,type:{name:"string"}}}}}catch{}const De={title:"Components/GiftRecommendationCard",component:_,parameters:{layout:"centered"},tags:["autodocs"],argTypes:{onShowDetails:{action:"show details clicked"},onSelect:{action:"selection toggled"}}},a={gift:{id:"1",name:"Premium Kahve Makinesi",price:2499.99,image_url:"https://via.placeholder.com/400",category:"Ev & Yaşam",rating:4.5,availability:!0},reasoning:["Kullanıcının hobi listesinde kahve yapımı var, bu ürün mükemmel bir eşleşme","Bütçe aralığına uygun fiyat","Yaş grubuna uygun sofistike bir hediye"],confidence:.85},n={review_analysis:{average_rating:4.5,review_count:1234},trend_analysis:{trending:!0,trend_score:.8},inventory_check:{available:!0,stock_count:50}},g={args:{recommendation:a,toolResults:n}},p={args:{recommendation:{...a,confidence:.92},toolResults:n}},f={args:{recommendation:{...a,confidence:.65},toolResults:n}},x={args:{recommendation:{...a,confidence:.35},toolResults:n}},h={args:{recommendation:{...a,reasoning:["Kullanıcının hobi listesinde kahve yapımı var, bu ürün mükemmel bir eşleşme","Bütçe aralığına uygun fiyat, kullanıcının belirlediği maksimum bütçenin altında","Yaş grubuna uygun sofistike bir hediye, 30-40 yaş arası kullanıcılar için ideal","Yüksek kaliteli malzeme ve dayanıklılık özellikleri","Kolay kullanım ve temizlik imkanı","Enerji tasarruflu model"]},toolResults:n}},k={args:{recommendation:a}},b={args:{recommendation:a,toolResults:n,isSelected:!0}},y={args:{recommendation:a,toolResults:n,isSelected:!1}},v={args:{recommendation:{...a,reasoning:["Hobi eşleşmesi mükemmel"]},toolResults:n}},R={args:{recommendation:{...a,gift:{...a.gift,image_url:void 0}},toolResults:n}},j={args:{recommendation:a,toolResults:{review_analysis:{average_rating:4.2,review_count:500}}}};var E,D,L;g.parameters={...g.parameters,docs:{...(E=g.parameters)==null?void 0:E.docs,source:{originalSource:`{
  args: {
    recommendation: mockRecommendation,
    toolResults: mockToolResults
  }
}`,...(L=(D=g.parameters)==null?void 0:D.docs)==null?void 0:L.source}}};var I,M,q;p.parameters={...p.parameters,docs:{...(I=p.parameters)==null?void 0:I.docs,source:{originalSource:`{
  args: {
    recommendation: {
      ...mockRecommendation,
      confidence: 0.92
    },
    toolResults: mockToolResults
  }
}`,...(q=(M=p.parameters)==null?void 0:M.docs)==null?void 0:q.source}}};var K,B,G;f.parameters={...f.parameters,docs:{...(K=f.parameters)==null?void 0:K.docs,source:{originalSource:`{
  args: {
    recommendation: {
      ...mockRecommendation,
      confidence: 0.65
    },
    toolResults: mockToolResults
  }
}`,...(G=(B=f.parameters)==null?void 0:B.docs)==null?void 0:G.source}}};var Y,V,W;x.parameters={...x.parameters,docs:{...(Y=x.parameters)==null?void 0:Y.docs,source:{originalSource:`{
  args: {
    recommendation: {
      ...mockRecommendation,
      confidence: 0.35
    },
    toolResults: mockToolResults
  }
}`,...(W=(V=x.parameters)==null?void 0:V.docs)==null?void 0:W.source}}};var $,H,P;h.parameters={...h.parameters,docs:{...($=h.parameters)==null?void 0:$.docs,source:{originalSource:`{
  args: {
    recommendation: {
      ...mockRecommendation,
      reasoning: ['Kullanıcının hobi listesinde kahve yapımı var, bu ürün mükemmel bir eşleşme', 'Bütçe aralığına uygun fiyat, kullanıcının belirlediği maksimum bütçenin altında', 'Yaş grubuna uygun sofistike bir hediye, 30-40 yaş arası kullanıcılar için ideal', 'Yüksek kaliteli malzeme ve dayanıklılık özellikleri', 'Kolay kullanım ve temizlik imkanı', 'Enerji tasarruflu model']
    },
    toolResults: mockToolResults
  }
}`,...(P=(H=h.parameters)==null?void 0:H.docs)==null?void 0:P.source}}};var A,O,J;k.parameters={...k.parameters,docs:{...(A=k.parameters)==null?void 0:A.docs,source:{originalSource:`{
  args: {
    recommendation: mockRecommendation
  }
}`,...(J=(O=k.parameters)==null?void 0:O.docs)==null?void 0:J.source}}};var Q,U,X;b.parameters={...b.parameters,docs:{...(Q=b.parameters)==null?void 0:Q.docs,source:{originalSource:`{
  args: {
    recommendation: mockRecommendation,
    toolResults: mockToolResults,
    isSelected: true
  }
}`,...(X=(U=b.parameters)==null?void 0:U.docs)==null?void 0:X.source}}};var Z,ee,ae;y.parameters={...y.parameters,docs:{...(Z=y.parameters)==null?void 0:Z.docs,source:{originalSource:`{
  args: {
    recommendation: mockRecommendation,
    toolResults: mockToolResults,
    isSelected: false
  }
}`,...(ae=(ee=y.parameters)==null?void 0:ee.docs)==null?void 0:ae.source}}};var ne,se,re;v.parameters={...v.parameters,docs:{...(ne=v.parameters)==null?void 0:ne.docs,source:{originalSource:`{
  args: {
    recommendation: {
      ...mockRecommendation,
      reasoning: ['Hobi eşleşmesi mükemmel']
    },
    toolResults: mockToolResults
  }
}`,...(re=(se=v.parameters)==null?void 0:se.docs)==null?void 0:re.source}}};var oe,te,ie;R.parameters={...R.parameters,docs:{...(oe=R.parameters)==null?void 0:oe.docs,source:{originalSource:`{
  args: {
    recommendation: {
      ...mockRecommendation,
      gift: {
        ...mockRecommendation.gift,
        image_url: undefined
      }
    },
    toolResults: mockToolResults
  }
}`,...(ie=(te=R.parameters)==null?void 0:te.docs)==null?void 0:ie.source}}};var le,ce,me;j.parameters={...j.parameters,docs:{...(le=j.parameters)==null?void 0:le.docs,source:{originalSource:`{
  args: {
    recommendation: mockRecommendation,
    toolResults: {
      review_analysis: {
        average_rating: 4.2,
        review_count: 500
      }
    }
  }
}`,...(me=(ce=j.parameters)==null?void 0:ce.docs)==null?void 0:me.source}}};const Le=["Default","HighConfidence","MediumConfidence","LowConfidence","LongReasoning","WithoutToolResults","Selected","WithSelectionCheckbox","MinimalReasoning","NoImage","PartialToolResults"];export{g as Default,p as HighConfidence,h as LongReasoning,x as LowConfidence,f as MediumConfidence,v as MinimalReasoning,R as NoImage,j as PartialToolResults,b as Selected,y as WithSelectionCheckbox,k as WithoutToolResults,Le as __namedExportsOrder,De as default};
