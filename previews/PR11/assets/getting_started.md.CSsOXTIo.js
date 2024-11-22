import{_ as t,c as a,a5 as i,o as s}from"./chunks/framework.dS0hJV0l.js";const u=JSON.parse('{"title":"Getting started with SpeciesDistributionModels.jl","description":"","frontmatter":{},"headers":[],"relativePath":"getting_started.md","filePath":"getting_started.md","lastUpdated":null}'),n={name:"getting_started.md"};function o(l,e,r,d,h,p){return s(),a("div",null,e[0]||(e[0]=[i('<h1 id="Getting-started-with-SpeciesDistributionModels.jl" tabindex="-1">Getting started with SpeciesDistributionModels.jl <a class="header-anchor" href="#Getting-started-with-SpeciesDistributionModels.jl" aria-label="Permalink to &quot;Getting started with SpeciesDistributionModels.jl {#Getting-started-with-SpeciesDistributionModels.jl}&quot;">​</a></h1><p>This package is under active development. Be advised that it may change at any time.</p><h2 id="installation" tabindex="-1">Installation <a class="header-anchor" href="#installation" aria-label="Permalink to &quot;Installation&quot;">​</a></h2><p>This package is not registered yet, but can easily be installed directly from GitHub.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] add github</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">com</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">tiemvanderdeure</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SpeciesDistributionModels</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">jl</span></span></code></pre></div><h2 id="Core-functionality" tabindex="-1">Core functionality <a class="header-anchor" href="#Core-functionality" aria-label="Permalink to &quot;Core functionality {#Core-functionality}&quot;">​</a></h2><p>A species distribution modelling workflow would typically consist of some data handling, then fitting a model, then evaluating it, and finally projecting to spatial data.</p><p>In this package, the main data handling tool is the <a href="/SpeciesDistributionModels.jl/previews/PR11/api#SpeciesDistributionModels.sdmdata-Tuple{Any, Any}">sdmdata</a> function, which takes two Tables.jl-compatible data objects (e.g. DataFrames) as input and returns an <code>SDMdata</code> object. You can also specify a resampling strategy or select a subset of variables in this step.</p><p>Next, this object and a <code>NamedTuple</code> of models is passed to the <a href="/SpeciesDistributionModels.jl/previews/PR11/api#SpeciesDistributionModels.sdm-Tuple{Any, Any}">sdm</a> function to fit the models. The models can be any object that implements the MLJ interface and is compatible with binary categorical data. See the <a href="https://juliaai.github.io/MLJ.jl/dev/model_browser/#Classification" target="_blank" rel="noreferrer">MLJ model registry</a> for a list of available models.</p><p>A fit ensemble can then be passed to functions like <code>SDM.evaluate</code> and <code>SDM.predict</code>.</p><h2 id="types" tabindex="-1">Types <a class="header-anchor" href="#types" aria-label="Permalink to &quot;Types&quot;">​</a></h2><p>Most types in this package are named acoording to the same nested structure with three levels; machine, group, and ensemble. This does for fitted models (<a href="./@ref">SDMmachine</a>, <a href="./@ref">SDMgroup</a>, and <a href="./@ref">SDMensemble</a>), evaluations of those models (<a href="./@ref">SDMmachineEvaluation</a> etc.), and mdoel explanations (<a href="./@ref">SDMmachineExplanation</a> etc.). The meaning of machine, group, and ensemble in this package is as follows:</p><ul><li><p>machine: a single instance of one particular model fit using one particular set of data.</p></li><li><p>group: one or more machines, which are instances of the same model, but may be fit on different sets of data (e.g. resampling folds)</p></li><li><p>ensemble: one or more groups, which each use different models.</p></li></ul>',13)]))}const m=t(n,[["render",o]]);export{u as __pageData,m as default};
